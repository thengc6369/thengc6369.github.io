/*
 * MDGM -> ACB-index, running entirely in the browser.
 * Version 2: interpolate the original 3600 x 1801 MDGM onto the NATIVE
 * 144 x 120 (usually) reference grid, then project onto two compressed
 * adjoint patterns implementing the original sequential ghost-EOF removal.
 * The coordinates are taken directly from blue_EOFanal.nc by the MATLAB
 * exporter; no approximation by image resizing is used.
 * Supports opaque 8-bit 3600x1801 RGB/RGBA PNGs.
 */
(() => {
  'use strict';
  const root = document.getElementById('acb-web-calculator');
  if (!root) return;
  const el = (id) => document.getElementById(id);
  const fileInput = el('acb-image-file');
  const runButton = el('acb-run');
  const status = el('acb-status');
  const results = el('acb-results');
  const downloadButton = el('acb-download');
  const base = (() => {
    // Jekyll baseurl is passed through the script src, so this also works
    // if the website later moves to a GitHub Pages project subpath.
    const scripts = document.querySelectorAll('script[src*="acb-calculator.js"]');
    const url = new URL(scripts[scripts.length - 1].src, document.baseURI);
    return new URL('../data/', url).href;
  })();
  let last = null;
  let modelPromise = null;

  const say = (message, error = false) => {
    status.textContent = message;
    status.classList.toggle('acb-error', error);
  };
  const format = (value) => Number(value).toFixed(5);

  function loadModel() {
    if (!modelPromise) {
      modelPromise = (async () => {
        const [configResponse, binaryResponse] = await Promise.all([
          fetch(base + 'acb-config.json'),
          fetch(base + 'acb-weights.bin')
        ]);
        if (!configResponse.ok || !binaryResponse.ok) {
          throw new Error('Reference files are missing (assets/data/acb-config.json and acb-weights.bin).');
        }
        const config = await configResponse.json();
        if (config.format !== 'ACB_WEB_WEIGHTS_V2' ||
            config.image_width !== 3600 || config.image_height !== 1801 ||
            !Number.isInteger(config.nlon) || !Number.isInteger(config.nlat) ||
            config.nlon < 2 || config.nlat < 2 ||
            !Array.isArray(config.longitude) || config.longitude.length !== config.nlon ||
            !Array.isArray(config.latitude) || config.latitude.length !== config.nlat ||
            !config.longitude.every(x => Number.isFinite(x) && x >= 0 && x < 360) ||
            !config.latitude.every(y => Number.isFinite(y) && y >= -90.000001 && y <= 90.000001) ||
            !Array.isArray(config.bias) || config.bias.length !== 2 ||
            !Array.isArray(config.pc_mean_raw) || config.pc_mean_raw.length !== 2 ||
            !Array.isArray(config.pc_std_raw) || config.pc_std_raw.length !== 2) {
          throw new Error('Invalid reference configuration. Regenerate it with the supplied MATLAB script.');
        }
        const buffer = await binaryResponse.arrayBuffer();
        const n = config.nlon * config.nlat;
        if (buffer.byteLength !== 2 * n * 4) {
          throw new Error('Reference binary has the wrong size.');
        }
        if (![...config.bias, ...config.pc_mean_raw, ...config.pc_std_raw,
              config.acb_mean_raw, config.acb_std_raw].every(Number.isFinite) ||
            config.acb_std_raw <= 0 || config.pc_std_raw.some(v => v <= 0)) {
          throw new Error('Invalid normalization constants in reference configuration.');
        }
        // MATLAB exporter writes IEEE float32 little endian, native NetCDF order
        // (longitude-fastest), with PC2 pattern followed by PC3 pattern.
        const weights = new Float32Array(buffer);
        return { config, w2: weights.subarray(0, n), w3: weights.subarray(n, 2*n) };
      })().catch(err => {
        modelPromise = null; // permit retry after a failed deployment
        throw err;
      });
    }
    return modelPromise;
  }

  async function decodePng(file, width, height) {
    // Decode PNG samples directly, without Canvas. Canvas can apply PNG
    // color/gamma profiles, silently changing scientific blue-channel bytes.
    if (!('DecompressionStream' in window)) {
      throw new Error('This browser cannot decode the reference PNG losslessly. Please use a recent Safari, Firefox or Chrome.');
    }
    const data = new Uint8Array(await file.arrayBuffer());
    const signature = [137,80,78,71,13,10,26,10];
    if (data.length < 33 || signature.some((v,i) => data[i] !== v)) {
      throw new Error('The selected file is not a valid PNG.');
    }
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    let pos = 8;
    let pngWidth = -1, pngHeight = -1, bpp = -1;
    let seenHeader = false, seenEnd = false;
    const idat = [];
    while (pos + 12 <= data.length) {
      const length = view.getUint32(pos, false);
      const end = pos + 12 + length;
      if (end > data.length) throw new Error('The PNG file is truncated.');
      const type = String.fromCharCode(data[pos+4], data[pos+5], data[pos+6], data[pos+7]);
      const start = pos + 8;
      if (type === 'IHDR') {
        if (seenHeader || length !== 13) throw new Error('Invalid PNG header.');
        seenHeader = true;
        pngWidth = view.getUint32(start, false);
        pngHeight = view.getUint32(start+4, false);
        const bitDepth = data[start+8];
        const colorType = data[start+9];
        if (pngWidth !== width || pngHeight !== height) {
          throw new Error(`MDGM must be exactly ${width} × ${height} pixels (received ${pngWidth} × ${pngHeight}).`);
        }
        if (bitDepth !== 8 || (colorType !== 2 && colorType !== 6) ||
            data[start+10] !== 0 || data[start+11] !== 0 || data[start+12] !== 0) {
          throw new Error('Use a noninterlaced, 8-bit RGB or RGBA PNG. Palette/grayscale/16-bit files are not supported.');
        }
        bpp = colorType === 2 ? 3 : 4;
      } else if (type === 'IDAT') {
        if (!seenHeader) throw new Error('Missing PNG header.');
        idat.push(data.subarray(start, start+length));
      } else if (type === 'IEND') {
        seenEnd = true;
        break;
      }
      pos = end;
    }
    if (!seenHeader || !seenEnd || !idat.length) throw new Error('Incomplete PNG file.');
    // PNG IDAT is a zlib-wrapped DEFLATE stream. "deflate" supports that
    // wrapper (as opposed to "deflate-raw").
    let raw;
    try {
      const blob = new Blob(idat);
      const stream = blob.stream().pipeThrough(new DecompressionStream('deflate'));
      raw = new Uint8Array(await new Response(stream).arrayBuffer());
    } catch (err) {
      throw new Error('Could not decompress PNG image data.');
    }
    const stride = width * bpp;
    if (raw.length !== height * (stride + 1)) {
      throw new Error('PNG image data have unexpected dimensions.');
    }
    const blue = new Uint8Array(width * height);
    let previous = new Uint8Array(stride), current = new Uint8Array(stride);
    let offset = 0;
    for (let y = 0; y < height; y++) {
      const filter = raw[offset++];
      if (filter > 4) throw new Error('Unsupported PNG scanline filter.');
      for (let i = 0; i < stride; i++) {
        const sample = raw[offset++];
        const left = i >= bpp ? current[i-bpp] : 0;
        const up = previous[i];
        const upperLeft = i >= bpp ? previous[i-bpp] : 0;
        let predictor = 0;
        if (filter === 1) predictor = left;
        else if (filter === 2) predictor = up;
        else if (filter === 3) predictor = Math.floor((left + up) / 2);
        else if (filter === 4) {
          const p = left + up - upperLeft;
          const a = Math.abs(p-left), b = Math.abs(p-up), c = Math.abs(p-upperLeft);
          predictor = a <= b && a <= c ? left : b <= c ? up : upperLeft;
        }
        current[i] = (sample + predictor) & 255;
      }
      for (let x = 0; x < width; x++) {
        const p = x*bpp;
        if (bpp === 4 && current[p+3] !== 255) {
          throw new Error('Transparent pixels are not supported. Use an opaque RGB MDGM PNG.');
        }
        blue[y*width + x] = current[p+2];
      }
      [previous,current] = [current,previous];
    }
    return blue;
  }

  function evaluate(blue, model) {
    const {config, w2, w3} = model;
    const { image_width: width, image_height: height, nlon, nlat } = config;
    let s2 = 0, s3 = 0;
    // Reproduce the original MATLAB griddedInterpolant with periodic
    // longitude extension and linear interpolation in ascending latitude.
    // MDGM PNG is stored with longitude 0..359.9 from left to right and
    // latitude 90..-90 from top to bottom. No image resizing involved.
    for (let j = 0; j < nlat; j++) {
      const y = Math.max(0, Math.min(height - 1, (90 - config.latitude[j]) * 10));
      const y0 = Math.floor(y);
      const y1 = Math.min(height - 1, y0 + 1);
      const fy = y - y0;
      const row0 = y0 * width, row1 = y1 * width;
      for (let i = 0; i < nlon; i++) {
        const x = config.longitude[i] * 10;
        const xFloor = Math.floor(x);
        const x0 = xFloor % width;
        const x1 = (x0 + 1) % width;
        const fx = x - xFloor;
        const top = (1-fx)*blue[row0+x0] + fx*blue[row0+x1];
        const bot = (1-fx)*blue[row1+x0] + fx*blue[row1+x1];
        const b = (1-fy)*top + fy*bot;
        const k = j*nlon + i;  // native MATLAB/NetCDF lon-fastest order
        s2 += b * w2[k];
        s3 += b * w3[k];
      }
    }
    const pc2raw = s2 + config.bias[0];
    const pc3raw = s3 + config.bias[1];
    const amplitude = Math.hypot(pc2raw, pc3raw);
    if (!Number.isFinite(amplitude) || amplitude <= 0) {
      throw new Error('Could not calculate a finite, nonzero PC amplitude.');
    }
    const normalizedAmplitude = (amplitude - config.acb_mean_raw) / config.acb_std_raw;
    const acb1 = pc2raw / amplitude * normalizedAmplitude;
    const acb2 = pc3raw / amplitude * normalizedAmplitude;
    return {
      acb1, acb2, pc2raw, pc3raw,
      pc2: (pc2raw - config.pc_mean_raw[0]) / config.pc_std_raw[0],
      pc3: (pc3raw - config.pc_mean_raw[1]) / config.pc_std_raw[1]
    };
  }

  function drawPlot(x, y) {
    const canvas = el('acb-phase-plot');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const cx = W/2, cy = H/2, radius = Math.min(W,H)*.37;
    const maxValue = Math.max(1, Math.abs(x), Math.abs(y)) * 1.12;
    const scale = radius / maxValue;
    ctx.lineWidth = 1; ctx.strokeStyle = '#b9c2ca';
    for (const f of [.25,.5,.75,1]) {
      ctx.beginPath(); ctx.arc(cx, cy, f*radius, 0, Math.PI*2); ctx.stroke();
    }
    ctx.strokeStyle = '#8d989f';
    ctx.beginPath(); ctx.moveTo(cx-radius-15, cy); ctx.lineTo(cx+radius+15, cy);
    ctx.moveTo(cx,cy-radius-15); ctx.lineTo(cx,cy+radius+15); ctx.stroke();
    ctx.fillStyle = '#555f67'; ctx.font = '13px sans-serif';
    ctx.fillText('ACB 1', cx+radius-42, cy-8);
    ctx.fillText('ACB 2', cx+10, cy-radius+5);
    ctx.fillText(`±${format(maxValue)}`, 8, H-14);
    const px = cx + x*scale, py = cy - y*scale;
    ctx.strokeStyle = '#a96343'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(px,py); ctx.stroke();
    ctx.fillStyle = '#a96343';
    ctx.beginPath(); ctx.arc(px,py,5,0,Math.PI*2); ctx.fill();
  }

  fileInput.addEventListener('change', () => {
    last = null;
    results.hidden = true;
    const file = fileInput.files && fileInput.files[0];
    runButton.disabled = !file;
    say(file ? `Selected ${file.name}. Click Calculate ACB-index.` : 'Select an MDGM PNG to begin.');
  });

  runButton.addEventListener('click', async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    runButton.disabled = true;
    results.hidden = true;
    last = null;
    try {
      if (!file.name.toLowerCase().endsWith('.png')) {
        throw new Error('Please select a PNG file.');
      }
      say('Loading compact reference projections (about 138 KB)…');
      // Let the status render before starting expensive work.
      await new Promise(resolve => requestAnimationFrame(resolve));
      const model = await loadModel();
      say('Decoding the image…');
      await new Promise(resolve => requestAnimationFrame(resolve));
      const blue = await decodePng(file, model.config.image_width, model.config.image_height);
      say('Calculating the ACB-index…');
      await new Promise(resolve => requestAnimationFrame(resolve));
      const output = evaluate(blue, model);
      last = {file: file.name, ...output};
      el('acb-1').textContent = format(output.acb1);
      el('acb-2').textContent = format(output.acb2);
      el('acb-extra').textContent = `Normalized PC2: ${format(output.pc2)} · Normalized PC3: ${format(output.pc3)}`;
      drawPlot(output.acb1, output.acb2);
      results.hidden = false;
      say('Calculation complete.');
    } catch (error) {
      say(error && error.message ? error.message : 'Calculation failed.', true);
      console.error('[ACB calculator]', error);
    } finally {
      runButton.disabled = !(fileInput.files && fileInput.files.length);
    }
  });

  downloadButton.addEventListener('click', () => {
    if (!last) return;
    const csv = [
      'filename,ACB_component_1,ACB_component_2,PC2_normalized,PC3_normalized,PC2_raw,PC3_raw',
      [JSON.stringify(last.file), last.acb1, last.acb2, last.pc2, last.pc3,
        last.pc2raw, last.pc3raw].join(',')
    ].join('\r\n') + '\r\n';
    const url = URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const a = document.createElement('a');
    a.href = url; a.download = 'ACB_index_result.csv';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
})();

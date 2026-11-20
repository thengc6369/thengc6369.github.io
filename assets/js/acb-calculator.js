/*
 * MDGM -> ACB-index, running entirely in the browser.
 * Version 4: batch processing of 1-50 MDGMs, optional Ls labels,
 * consecutive-point connections, and optional five-image causal mean.
 * Interpolate the original 3600 x 1801 MDGM onto the NATIVE
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
  const fileList = el('acb-file-list');
  const tableBody = el('acb-results-rows');
  const modeInputs = Array.from(root.querySelectorAll('input[name="acb-label-mode"]'));
  const filterInputs = Array.from(root.querySelectorAll('input[name="acb-filter-mode"]'));
  const MAX_IMAGES = 50;
  const base = (() => {
    // Jekyll baseurl is passed through the script src, so this also works
    // if the website later moves to a GitHub Pages project subpath.
    const scripts = document.querySelectorAll('script[src*="acb-calculator.js"]');
    const url = new URL(scripts[scripts.length - 1].src, document.baseURI);
    return new URL('../data/', url).href;
  })();
  let last = null;
  let selectedFiles = [];
  let running = false;
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

  // A batch is displayed in the order in which the browser supplies the files.
  // The file selection is capped at 50; images are decoded one at a time.
  const lsValues = [];
  const selectedMode = () => root.querySelector('input[name="acb-label-mode"]:checked').value;
  const selectedFilter = () => root.querySelector('input[name="acb-filter-mode"]:checked').value;

  // Filter the final two-component ACB-index, NOT the raw principal components.
  // The first four points use the available 1, 2, 3 or 4 values. From point 5
  // onward every filtered point averages exactly five consecutive images.
  // All filtering follows the file selection order (never sorted by Ls).
  function applyFilter(rawEntries, filter) {
    if (filter !== 'unfiltered' && filter !== 'causal5') {
      throw new Error('Unrecognized ACB-index filter.');
    }
    return rawEntries.map((row, i) => {
      const start = filter === 'causal5' ? Math.max(0, i - 4) : i;
      const count = i - start + 1;
      let sum1 = 0, sum2 = 0;
      for (let j = start; j <= i; j++) {
        sum1 += rawEntries[j].acb1;
        sum2 += rawEntries[j].acb2;
      }
      return { ...row, plotAcb1: sum1 / count, plotAcb2: sum2 / count,
               windowCount: count };
    });
  }

  function showResults(rawEntries) {
    const plotted = applyFilter(rawEntries, selectedFilter());
    renderTable(plotted);
    drawPlot(plotted, selectedMode());
    last = { rawEntries, entries: plotted, mode: selectedMode(), filter: selectedFilter() };
    results.hidden = false;
  }

  function invalidate() {
    last = null;
    results.hidden = true;
    runButton.disabled = running || selectedFiles.length === 0 || selectedFiles.length > MAX_IMAGES;
  }

  function renderSelectedFiles() {
    fileList.replaceChildren();
    const showLs = selectedMode() === 'ls';
    fileList.hidden = !selectedFiles.length || selectedFiles.length > MAX_IMAGES;
    if (fileList.hidden) return;

    selectedFiles.forEach((file, i) => {
      const row = document.createElement('div');
      row.className = 'acb-file-row';
      const name = document.createElement('span');
      name.className = 'acb-file-name';
      name.textContent = `${i + 1}. ${file.name}`;
      row.appendChild(name);

      const lsLabel = document.createElement('label');
      lsLabel.className = 'acb-ls-label';
      lsLabel.hidden = !showLs;
      lsLabel.textContent = 'Ls (degrees): ';
      const input = document.createElement('input');
      input.className = 'acb-ls-input';
      input.type = 'number';
      input.min = '0';
      input.max = '360';
      input.step = 'any';
      input.placeholder = '0–360';
      input.setAttribute('aria-label', `Ls for image ${i + 1}: ${file.name}`);
      input.value = lsValues[i] ?? '';
      input.addEventListener('input', () => {
        lsValues[i] = input.value;
        invalidate();
      });
      lsLabel.appendChild(input);
      row.appendChild(lsLabel);
      fileList.appendChild(row);
    });
  }

  function validateSolarLongitudes() {
    if (selectedMode() !== 'ls') return selectedFiles.map(() => null);
    return selectedFiles.map((file, i) => {
      const raw = (lsValues[i] ?? '').trim();
      if (raw === '') throw new Error(`Enter Ls for image ${i + 1} (${file.name}).`);
      const value = Number(raw);
      if (!Number.isFinite(value) || value < 0 || value > 360) {
        throw new Error(`Ls for image ${i + 1} must be between 0 and 360 degrees.`);
      }
      return value;
    });
  }

  function drawPlot(entries, mode) {
    const canvas = el('acb-phase-plot');
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Your browser could not initialize the plot.');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const centerX = W * 0.48, centerY = H * 0.49;
    const halfSize = 189;
    const limit = Math.max(1, ...entries.flatMap(r => [Math.abs(r.plotAcb1), Math.abs(r.plotAcb2)])) * 1.20;
    const scale = halfSize / limit;
    const X = v => centerX + v * scale;
    const Y = v => centerY - v * scale;

    // Symmetric, equal-scale ACB 1 / ACB 2 axes, shared by all images.
    ctx.save();
    ctx.font = '12px sans-serif';
    ctx.lineWidth = 1;
    for (let t = -4; t <= 4; t++) {
      const value = limit * t / 4;
      const px = X(value), py = Y(value);
      ctx.strokeStyle = t === 0 ? '#65717b' : '#e1e5e8';
      ctx.beginPath();
      ctx.moveTo(px, Y(-limit)); ctx.lineTo(px, Y(limit));
      ctx.moveTo(X(-limit), py); ctx.lineTo(X(limit), py);
      ctx.stroke();
      if (t % 2 === 0) {
        ctx.fillStyle = '#545b61';
        ctx.textAlign = 'center';
        ctx.fillText(value.toFixed(2), px, Y(-limit) + 17);
        if (t !== 0) {
          ctx.textAlign = 'right';
          ctx.fillText(value.toFixed(2), X(-limit) - 9, py + 4);
        }
      }
    }
    ctx.fillStyle = '#35404a';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('ACB component 1', centerX, H - 16);
    ctx.save();
    ctx.translate(22, centerY);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('ACB component 2', 0, 0);
    ctx.restore();

    // Draw a single light-blue polyline connecting consecutive images,
    // then draw dark-blue markers on top so all points remain visible.
    if (entries.length > 1) {
      ctx.beginPath();
      entries.forEach((row, i) => {
        const px = X(row.plotAcb1), py = Y(row.plotAcb2);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = '#8bc5e8'; // light blue
      ctx.lineWidth = 2.4;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.stroke();
    }
    for (const row of entries) {
      const px = X(row.plotAcb1), py = Y(row.plotAcb2);
      ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI);
      ctx.fillStyle = '#164a80'; // dark blue
      ctx.fill();
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Prefer nonoverlapping labels when points are clustered. The result
    // table below always provides the exact point/filename association.
    const placed = [];
    ctx.font = '12px sans-serif';
    ctx.textBaseline = 'middle';
    for (const row of entries) {
      const label = mode === 'ls' ? `Ls ${Number(row.ls).toLocaleString('en-US', {maximumFractionDigits: 2})}°` : String(row.number);
      const px = X(row.plotAcb1), py = Y(row.plotAcb2);
      const textWidth = ctx.measureText(label).width;
      const proposals = [
        [px+9, py-11], [px+9, py+12], [px-textWidth-9, py-11], [px-textWidth-9, py+12],
        [px-textWidth/2, py-18], [px-textWidth/2, py+19],
        [px+13, py], [px-textWidth-13, py]
      ];
      let chosen = null, bestPenalty = Infinity;
      for (const [tx, ty] of proposals) {
        const box = {left:tx-2, right:tx+textWidth+2, top:ty-9, bottom:ty+9};
        let penalty = 0;
        if (box.left < 0 || box.right > W || box.top < 0 || box.bottom > H) penalty += 1e6;
        for (const b of placed) {
          const dx = Math.max(0, Math.min(box.right,b.right)-Math.max(box.left,b.left));
          const dy = Math.max(0, Math.min(box.bottom,b.bottom)-Math.max(box.top,b.top));
          penalty += dx*dy;
        }
        if (penalty < bestPenalty) { chosen = {tx,ty,box}; bestPenalty = penalty; }
      }
      placed.push(chosen.box);
      ctx.textAlign = 'left';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3.5;
      ctx.strokeText(label, chosen.tx, chosen.ty);
      ctx.fillStyle = '#164a80';
      ctx.fillText(label, chosen.tx, chosen.ty);
    }
    ctx.restore();
  }

  function renderTable(entries) {
    tableBody.replaceChildren();
    for (const row of entries) {
      const tr = document.createElement('tr');
      const values = [row.number, row.file, row.ls === null ? '—' : row.ls,
        format(row.plotAcb1), format(row.plotAcb2),
        format(row.acb1), format(row.acb2), format(row.pc2), format(row.pc3)];
      for (const value of values) {
        const td = document.createElement('td');
        td.textContent = String(value);
        tr.appendChild(td);
      }
      tableBody.appendChild(tr);
    }
  }

  fileInput.addEventListener('change', () => {
    selectedFiles = Array.from(fileInput.files ?? []);
    lsValues.length = 0;
    renderSelectedFiles();
    invalidate();
    if (selectedFiles.length > MAX_IMAGES) {
      say(`You selected ${selectedFiles.length} images. Please select at most ${MAX_IMAGES} MDGM PNGs.`, true);
    } else if (selectedFiles.length) {
      say(`Selected ${selectedFiles.length} image(s). Choose the point labels and click Calculate ACB-index.`);
    } else {
      say('Select 1–50 MDGM PNG files to begin.');
    }
  });

  for (const radio of modeInputs) {
    radio.addEventListener('change', () => {
      renderSelectedFiles(); // Retains any Ls values already entered.
      invalidate();
      say(selectedFiles.length ? 'Point-label option changed. Calculate to update the diagram.' : 'Select 1–50 MDGM PNG files to begin.');
    });
  }

  // Switching filters updates the existing results without decoding any PNG again.
  for (const radio of filterInputs) {
    radio.addEventListener('change', () => {
      if (running) return;
      if (last) {
        showResults(last.rawEntries);
        say(selectedFilter() === 'causal5'
          ? 'Showing 5-image causal mean (initial points use available history).'
          : 'Showing unfiltered ACB-index values.');
      } else if (selectedFiles.length) {
        say('Filter selected. Click Calculate ACB-index to see the results.');
      }
    });
  }

  runButton.addEventListener('click', async () => {
    if (running || selectedFiles.length < 1 || selectedFiles.length > MAX_IMAGES) return;
    let ls;
    try {
      ls = validateSolarLongitudes();
      for (let i = 0; i < selectedFiles.length; i++) {
        if (!selectedFiles[i].name.toLowerCase().endsWith('.png')) {
          throw new Error(`Image ${i + 1} (${selectedFiles[i].name}) is not a PNG.`);
        }
      }
    } catch (error) {
      say(error.message, true);
      return;
    }
    running = true;
    runButton.disabled = true;
    fileInput.disabled = true;
    modeInputs.forEach(r => { r.disabled = true; });
    filterInputs.forEach(r => { r.disabled = true; });
    root.querySelectorAll('.acb-ls-input').forEach(input => { input.disabled = true; });
    results.hidden = true;
    last = null;
    try {
      say('Loading reference projections…');
      await new Promise(resolve => requestAnimationFrame(resolve));
      const model = await loadModel();
      const entries = [];
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        say(`Processing MDGM ${i + 1}/${selectedFiles.length}: ${file.name}`);
        await new Promise(resolve => requestAnimationFrame(resolve));
        try {
          const blue = await decodePng(file, model.config.image_width, model.config.image_height);
          const output = evaluate(blue, model);
          entries.push({number:i+1, file:file.name, ls:ls[i], ...output});
        } catch (error) {
          throw new Error(`Image ${i + 1} (${file.name}): ${error.message}`);
        }
      }
      showResults(entries);
      say(`Calculation complete: ${entries.length} MDGM(s). ` +
          (selectedFilter() === 'causal5' ? 'Showing 5-image causal mean.' : 'Showing unfiltered values.'));
    } catch (error) {
      say(error && error.message ? error.message : 'Calculation failed.', true);
      console.error('[ACB calculator]', error);
    } finally {
      running = false;
      fileInput.disabled = false;
      modeInputs.forEach(r => { r.disabled = false; });
      filterInputs.forEach(r => { r.disabled = false; });
      root.querySelectorAll('.acb-ls-input').forEach(input => { input.disabled = false; });
      invalidateButtonOnly();
    }
  });

  function invalidateButtonOnly() {
    runButton.disabled = selectedFiles.length === 0 || selectedFiles.length > MAX_IMAGES;
  }

  downloadButton.addEventListener('click', () => {
    if (!last) return;
    const quote = value => `"${String(value).replace(/"/g, '""')}"`;
    const lines = [
      'image_number,filename,Ls_degrees,filter,window_count,ACB_component_1_plotted,ACB_component_2_plotted,ACB_component_1_unfiltered,ACB_component_2_unfiltered,PC2_normalized,PC3_normalized,PC2_raw,PC3_raw'
    ];
    for (const row of last.entries) {
      lines.push([
        row.number, quote(row.file), row.ls === null ? '' : row.ls,
        last.filter, row.windowCount, row.plotAcb1, row.plotAcb2,
        row.acb1, row.acb2, row.pc2, row.pc3, row.pc2raw, row.pc3raw
      ].join(','));
    }
    const csv = lines.join('\r\n') + '\r\n';
    const url = URL.createObjectURL(new Blob([csv], {type:'text/csv;charset=utf-8'}));
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ACB_index_batch_results.csv';
    document.body.appendChild(a);
    a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
})();

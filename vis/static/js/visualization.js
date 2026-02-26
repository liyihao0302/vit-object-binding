async function loadJSON(path) {
    const response = await fetch(path);
    return response.json();
}

async function loadImage(path) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = (e) => reject(e);
        img.src = path;
    });
}

/******************************
 *     JET colormap
 ******************************/
function jetColor(v) {
    v = Math.min(Math.max(v, 0), 1);

    let r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 3)));
    let g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 2)));
    let b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 1)));

    return [
        Math.floor(r * 255),
        Math.floor(g * 255),
        Math.floor(b * 255)
    ];
}

async function main() {

    /***********************************************
     * 1. Setup: we have 5 images
     ***********************************************/
    const IMAGE_IDS = ["img1", "img2", "img3", "img4", "img5"];

    const c0 = document.getElementById("canvas-original");
    const ctx0 = c0.getContext("2d");

    const c1 = document.getElementById("canvas-overlay");
    const ctx1 = c1.getContext("2d");

    const layerSlider = document.getElementById("layer-slider");
    const layerLabel  = document.getElementById("layer-label");

    const prevBtn = document.getElementById("prev-image");
    const nextBtn = document.getElementById("next-image");
    const imageLabel = document.getElementById("image-label");

    let currentImageIndex = 0;
    let currentImageId = IMAGE_IDS[currentImageIndex];

    let layerIds = [];          // e.g., [0, 3, ..., 23]
    let issameAllLayers = null; // [numLayers][1369][1369]

    let img = null;
    let patchedMask = null;

    let numPatchesH = 0;
    let numPatchesW = 0;
    let numPatches = 0;
    let patchSize = 0;

    let validPatches = [];
    let selectedPatchIdx = null;
    let selectedPatchCoord = null;

    // default layer = 18
    let currentLayerIndex = 0;

    function updateImageLabel() {
        imageLabel.textContent = `Image ${currentImageIndex + 1} / ${IMAGE_IDS.length}`;
    }

    /***********************************************
     * 2. Load everything for one image
     ***********************************************/
    async function loadDataForImage(imageId) {
        console.log("Loading:", imageId);
        const basePath = `../data/${imageId}`;

        img = await loadImage(`${basePath}/rgb_image.png`);
        const issameRaw = await loadJSON(`${basePath}/issameobject.json`);
        patchedMask = await loadJSON(`${basePath}/patched_mask.json`);

        // ---- Layer info ----
        if (layerIds.length === 0) {
            layerIds = issameRaw.layers.slice();

            // default = layer 18
            currentLayerIndex = layerIds.indexOf(18);
            if (currentLayerIndex === -1) currentLayerIndex = 0;

            setupLayerSlider();
        }

        issameAllLayers = issameRaw.issame;

        numPatchesH = patchedMask.length;
        numPatchesW = patchedMask[0].length;
        numPatches = numPatchesH * numPatchesW;

        // Canvas sizes = real image size
        c0.width = img.width;
        c0.height = img.height;
        c1.width = img.width;
        c1.height = img.height;

        patchSize = img.width / numPatchesW;

        // ---- valid patches ----
        validPatches = [];
        for (let py = 0; py < numPatchesH; py++) {
            for (let px = 0; px < numPatchesW; px++) {
                if (patchedMask[py][px]) {
                    validPatches.push([py, px]);
                }
            }
        }

        if (validPatches.length > 0) {
            const [py0, px0] = validPatches[Math.floor(Math.random() * validPatches.length)];
            selectedPatchCoord = [py0, px0];
            selectedPatchIdx = py0 * numPatchesW + px0;
        }

        updateImageLabel();
        renderLeft();
        renderOverlay();
    }

    /***********************************************
     * 3. Setup layer slider (JET heatmap)
     ***********************************************/
    function setupLayerSlider() {
        layerSlider.min = 0;
        layerSlider.max = layerIds.length - 1;
        layerSlider.step = 1;

        layerSlider.value = currentLayerIndex;
        layerLabel.textContent = layerIds[currentLayerIndex];

        layerSlider.addEventListener("input", () => {
            currentLayerIndex = parseInt(layerSlider.value, 10);
            layerLabel.textContent = layerIds[currentLayerIndex];
            renderOverlay();
        });
    }

    /***********************************************
     * 4. Render: LEFT (original + red box)
     ***********************************************/
    function renderLeft() {
        ctx0.clearRect(0, 0, c0.width, c0.height);
        ctx0.drawImage(img, 0, 0, img.width, img.height);

        if (selectedPatchCoord) {
            const [pySel, pxSel] = selectedPatchCoord;
            ctx0.strokeStyle = "red";
            ctx0.lineWidth = 3;
            ctx0.strokeRect(
                pxSel * patchSize,
                pySel * patchSize,
                patchSize,
                patchSize
            );
        }
    }
    function drawColorbar() {
        const cb = document.getElementById("canvas-colorbar");
        const cbCtx = cb.getContext("2d");

        const barWidth = 20;
        const barHeight = Math.floor(c1.height * 0.6);

        // Canvas size: width = bar + labels + vertical title space
        cb.width = barWidth + 60;       // 20 bar + 40 for labels/title
        cb.height = barHeight + 40;     // padding top + bottom

        const w = cb.width;
        const h = cb.height;

        const x0 = 20;    // bar left x
        const y0 = 20;    // bar top y

        const steps = 100;

        cbCtx.clearRect(0, 0, w, h);

        /***********************
         * 1. Draw colorbar
         ***********************/
        for (let i = 0; i < steps; i++) {
            const v = i / (steps - 1);
            const [r, g, b] = jetColor(v);

            const y = y0 + (barHeight - (i / (steps - 1)) * barHeight);
            cbCtx.fillStyle = `rgb(${r},${g},${b})`;
            cbCtx.fillRect(x0, y, barWidth, barHeight / steps);
        }

        // Border
        cbCtx.strokeStyle = "black";
        cbCtx.lineWidth = 1;
        cbCtx.strokeRect(x0, y0, barWidth, barHeight);

        /***********************
         * 2. Horizontal labels
         ***********************/
        cbCtx.fillStyle = "black";
        cbCtx.font = "11px sans-serif";
        cbCtx.textAlign = "left";
        cbCtx.textBaseline = "middle";

        cbCtx.fillText("1.0", x0 + barWidth + 6, y0 + 4);
        cbCtx.fillText("0.0", x0 + barWidth + 6, y0 + barHeight - 4);

        /***********************
         * 3. Vertical title
         *    (rotated 90 degrees)
         ***********************/
        cbCtx.save();

        // Move origin to right of bar center
        const titleX = x0 + barWidth + 20;           // horizontal offset beside bar
        const titleY = y0 + barHeight / 2;           // vertical center

        cbCtx.translate(titleX, titleY);
        cbCtx.rotate(-Math.PI / 2);                  // rotate 90° clockwise

        cbCtx.font = "italic 16px sans-serif";
        cbCtx.textAlign = "center";
        cbCtx.textBaseline = "middle";
        cbCtx.fillText("IsSameObject", 0, 0);

        cbCtx.restore();
    }





    /***********************************************
     * 5. Render: RIGHT (JET heatmap)
     ***********************************************/
    function renderOverlay() {
        ctx1.clearRect(0, 0, c1.width, c1.height);

        if (!issameAllLayers || selectedPatchIdx === null) {
            drawColorbar();
            return;
        }

        const layerMatrix = issameAllLayers[currentLayerIndex];
        const row = layerMatrix[selectedPatchIdx];

        const alpha = 0.5;  // overlay strength

        // 1) Draw original image as background
        ctx1.drawImage(img, 0, 0, img.width, img.height);

        // 2) Offscreen low-res heatmap: one pixel per patch
        const off = document.createElement("canvas");
        off.width = numPatchesW;   // 37
        off.height = numPatchesH;  // 37
        const offCtx = off.getContext("2d");

        const imgData = offCtx.createImageData(numPatchesW, numPatchesH);
        const data = imgData.data;

        for (let py = 0; py < numPatchesH; py++) {
            for (let px = 0; px < numPatchesW; px++) {
                const idx = py * numPatchesW + px;
                const di = 4 * idx;

                if (!patchedMask[py][px]) {
                    // invalid patch → fully transparent
                    data[di + 0] = 0;
                    data[di + 1] = 0;
                    data[di + 2] = 0;
                    data[di + 3] = 0;
                    continue;
                }

                const v = row[idx];           // [0,1]
                const [r, g, b] = jetColor(v);

                data[di + 0] = r;
                data[di + 1] = g;
                data[di + 2] = b;
                data[di + 3] = Math.floor(alpha * 255);  // global alpha
            }
        }

        offCtx.putImageData(imgData, 0, 0);

        // 3) Upscale heatmap ONLY to the patch grid area
        const heatmapWidth = numPatchesW * patchSize;   // should be 37 * 14 = 518
        const heatmapHeight = numPatchesH * patchSize;  // same

        ctx1.imageSmoothingEnabled = true;
        ctx1.imageSmoothingQuality = "high";

        ctx1.drawImage(
            off,
            0, 0, off.width, off.height,       // source: 37×37
            0, 0, heatmapWidth, heatmapHeight  // dest: exactly patch grid region
        );

        // 4) Selected patch red box (aligned with 14×14 grid)
        if (selectedPatchCoord) {
            const [pySel, pxSel] = selectedPatchCoord;
            ctx1.strokeStyle = "red";
            ctx1.lineWidth = 3;
            ctx1.strokeRect(
                pxSel * patchSize,
                pySel * patchSize,
                patchSize,
                patchSize
            );
        }

        // 5) Colorbar
        drawColorbar();
    }


    /***********************************************
     * 6. Click handler (choose patch)
     ***********************************************/
    c0.addEventListener("click", (event) => {
        const rect = c0.getBoundingClientRect();
        const xClick = event.clientX - rect.left;
        const yClick = event.clientY - rect.top;

        const px = Math.floor(xClick / patchSize);
        const py = Math.floor(yClick / patchSize);

        if (px < 0 || px >= numPatchesW || py < 0 || py >= numPatchesH) return;
        if (!patchedMask[py][px]) return;

        selectedPatchCoord = [py, px];
        selectedPatchIdx = py * numPatchesW + px;

        renderLeft();

        renderOverlay();
        drawColorbar();
        
    });

    /***********************************************
     * 7. Left/Right image switching
     ***********************************************/
    prevBtn.addEventListener("click", async () => {
        currentImageIndex = (currentImageIndex - 1 + IMAGE_IDS.length) % IMAGE_IDS.length;
        currentImageId = IMAGE_IDS[currentImageIndex];
        await loadDataForImage(currentImageId);
    });

    nextBtn.addEventListener("click", async () => {
        currentImageIndex = (currentImageIndex + 1) % IMAGE_IDS.length;
        currentImageId = IMAGE_IDS[currentImageIndex];
        await loadDataForImage(currentImageId);
    });

    /***********************************************
     * 8. Initial load
     ***********************************************/
    await loadDataForImage(currentImageId);

    console.log("Visualization ready!");
}

window.onload = main;

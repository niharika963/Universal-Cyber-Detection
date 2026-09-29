"use strict";

const API_URL = "";

let selectedFile = null;
let csvHeaders = [];
let csvRows = [];
let rocChart = null;
let lastAnalysis = null;

/* Prevent accidental form submission */
document.addEventListener(
    "submit",
    function (event) {
        event.preventDefault();
        event.stopImmediatePropagation();
    },
    true
);

console.log("========================================");
console.log("UNIVERSAL CYBER DETECTION SYSTEM");
console.log("NEW FIXED APP.JS LOADED");
console.log("========================================");


/* =========================================================
   KDD CUP 99 FEATURES
   ========================================================= */

const REQUIRED_FEATURES = [

    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",

    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",

    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",

    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"

];


const CATEGORICAL_FEATURES = [
    "protocol_type",
    "service",
    "flag"
];


const LABEL_COLUMNS = [
    "label",
    "class",
    "attack",
    "attack_type",
    "connection_type",
    "target",
    "category"
];


/* =========================================================
   PAGE START
   ========================================================= */
   document.addEventListener(
    "submit",
    function (event) {
        event.preventDefault();
        event.stopPropagation();
    },
    true
);

document.addEventListener("DOMContentLoaded", function () {

    console.log("DOM READY");

    const fileInput = document.getElementById("file-input");
    const detectButton = document.getElementById("detect-button");
    const evaluateButton = document.getElementById("evaluate-button");
    const analysisButton = document.getElementById("analysis-button");

    console.log("File input:", fileInput);
    console.log("Detect button:", detectButton);
    console.log("Evaluate button:", evaluateButton);
    console.log("Analysis button:", analysisButton);

    if (fileInput) {
        fileInput.addEventListener("change", handleFileSelection);
    }

    if (detectButton) {
        detectButton.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();

            console.log("DETECT ATTACK CLICKED");

            detectAttack();
        });
    }

    if (evaluateButton) {
        evaluateButton.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();

            console.log("========================================");
            console.log("EVALUATE DATASET CLICKED");
            console.log("Selected file:", selectedFile);
            console.log("========================================");

            evaluateDataset();
        });
    }

    if (analysisButton) {
        analysisButton.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();

            console.log("RUN COMPLETE ANALYSIS CLICKED");

            runCompleteAnalysis();
        });
    }

    disableButtons();

    checkBackend();

});
/* =========================================================
   BACKEND CHECK
   ========================================================= */

async function checkBackend() {

    setText(
        "backend-status",
        "Checking..."
    );


    const indicator =
        document.getElementById(
            "backend-indicator"
        );


    if (indicator) {

        indicator.className =
            "status-indicator checking";

    }


    try {

        const response =
            await fetchWithTimeout(
                API_URL + "/health",
                {
                    cache: "no-store"
                },
                10000
            );


        const data =
            await readJSON(response);


        if (!response.ok) {

            throw new Error(
                extractBackendError(data)
            );

        }


        setText(
            "backend-status",
            "Backend Connected"
        );


        if (indicator) {

            indicator.className =
                "status-indicator online";

        }


        addLog(
            "ONLINE",
            "FastAPI backend connected successfully."
        );


        console.log(
            "Backend health:",
            data
        );


    } catch (error) {

        setText(
            "backend-status",
            "Backend Offline"
        );


        if (indicator) {

            indicator.className =
                "status-indicator offline";

        }


        addLog(
            "ERROR",
            "Backend connection failed: " +
            error.message
        );


        console.error(
            "Backend check failed:",
            error
        );

    }

}


/* =========================================================
   FETCH WITH TIMEOUT
   ========================================================= */

async function fetchWithTimeout(
    url,
    options = {},
    timeoutMs = 120000
) {

    const controller =
        new AbortController();


    const timer =
        setTimeout(
            () => controller.abort(),
            timeoutMs
        );


    try {

        return await fetch(
            url,
            {
                ...options,
                signal: controller.signal
            }
        );

    } finally {

        clearTimeout(timer);

    }

}


/* =========================================================
   READ JSON
   ========================================================= */

async function readJSON(response) {

    const text =
        await response.text();


    if (!text) {

        return {};

    }


    try {

        return JSON.parse(text);

    } catch {

        return {
            detail: text
        };

    }

}


/* =========================================================
   FILE SELECTION
   ========================================================= */

async function handleFileSelection(event) {

    console.log("========================================");
    console.log("FILE SELECTION EVENT FIRED");
    console.log("========================================");

    const input = event.target;

    if (!input.files || input.files.length === 0) {
        console.log("No file selected.");

        selectedFile = null;

        setText(
            "file-status",
            "No dataset selected"
        );

        disableButtons();
        resetDatasetInfo();

        return;
    }

    selectedFile = input.files[0];

    console.log(
        "Selected file:",
        selectedFile.name
    );

    console.log(
        "File size:",
        selectedFile.size,
        "bytes"
    );

    if (
        !selectedFile.name
            .toLowerCase()
            .endsWith(".csv")
    ) {

        alert("Please select a CSV file.");

        selectedFile = null;
        input.value = "";

        disableButtons();

        return;
    }

    setText(
        "file-status",
        "Reading " + selectedFile.name + "..."
    );

    try {

        const fileText =
            await selectedFile.text();

        console.log(
            "CSV characters:",
            fileText.length
        );

        const parsed =
            parseCSV(fileText);

        csvHeaders =
            parsed.headers;

        csvRows =
            parsed.rows;

        console.log(
            "CSV headers:",
            csvHeaders
        );

        console.log(
            "CSV rows:",
            csvRows.length
        );

        if (!csvRows.length) {
            throw new Error(
                "The CSV file contains no data rows."
            );
        }

        updateDatasetInfo();

        const compatibility =
            checkCompatibility();

        displayCompatibility(
            compatibility
        );

        setText(
            "file-status",
            selectedFile.name +
            " loaded successfully — " +
            csvRows.length.toLocaleString() +
            " records"
        );

        /* Evaluation and complete analysis
           work with uploaded CSV datasets. */

        const evaluateButton =
            document.getElementById(
                "evaluate-button"
            );

        const analysisButton =
            document.getElementById(
                "analysis-button"
            );

        if (evaluateButton) {
            evaluateButton.disabled = false;
        }

        if (analysisButton) {
            analysisButton.disabled = false;
        }

        const detectButton =
            document.getElementById(
                "detect-button"
            );

        if (detectButton) {
            detectButton.disabled =
                !compatibility.compatible;
        }

        hideElement(
            "detection-section"
        );

        hideElement(
            "evaluation-section"
        );

        hideElement(
            "analysis-section"
        );

        addLog(
            "UPLOAD",
            selectedFile.name +
            " loaded successfully."
        );

        addLog(
            "DATASET",
            csvRows.length.toLocaleString() +
            " records detected."
        );

        addLog(
            "READY",
            "Dataset is ready for evaluation."
        );

        console.log(
            "DATASET READY FOR EVALUATION"
        );

    } catch (error) {

        console.error(
            "CSV loading error:",
            error
        );

        selectedFile = null;

        disableButtons();

        setText(
            "file-status",
            "Failed to load dataset"
        );

        resetDatasetInfo();

        alert(
            "Could not read the CSV file.\n\n" +
            error.message
        );
    }
}

/* =========================================================
   CSV PARSER
   ========================================================= */

function parseCSV(text) {

    const records = [];

    let row = [];

    let value = "";

    let quoted = false;


    for (
        let i = 0;
        i < text.length;
        i++
    ) {

        const ch =
            text[i];

        const next =
            text[i + 1];


        if (
            ch === '"' &&
            quoted &&
            next === '"'
        ) {

            value += '"';

            i++;

        }

        else if (ch === '"') {

            quoted = !quoted;

        }

        else if (
            ch === "," &&
            !quoted
        ) {

            row.push(value);

            value = "";

        }

        else if (
            (ch === "\n" ||
             ch === "\r") &&
            !quoted
        ) {

            if (
                ch === "\r" &&
                next === "\n"
            ) {

                i++;

            }


            row.push(value);

            value = "";


            if (
                row.some(
                    v =>
                        String(v).trim() !== ""
                )
            ) {

                records.push(row);

            }


            row = [];

        }

        else {

            value += ch;

        }

    }


    if (
        value !== "" ||
        row.length
    ) {

        row.push(value);


        if (
            row.some(
                v =>
                    String(v).trim() !== ""
            )
        ) {

            records.push(row);

        }

    }


    if (!records.length) {

        return {
            headers: [],
            rows: []
        };

    }


    const headers =
        records[0].map(
            normalizeColumn
        );


    const rows =
        records
            .slice(1)
            .map(values => {

                const object = {};


                headers.forEach(
                    (header, index) => {

                        object[header] =
                            values[index] === undefined
                                ? ""
                                : String(
                                    values[index]
                                ).trim();

                    }
                );


                return object;

            });


    return {
        headers,
        rows
    };

}


/* =========================================================
   NORMALIZE COLUMN
   ========================================================= */

function normalizeColumn(value) {

    return String(value)

        .replace(
            /^\uFEFF/,
            ""
        )

        .trim()

        .toLowerCase()

        .replace(
            /\s+/g,
            "_"
        )

        .replace(
            /[-.]/g,
            "_"
        );

}


/* =========================================================
   COMPATIBILITY
   ========================================================= */

function checkCompatibility() {

    const available =
        new Set(csvHeaders);


    const missing =
        REQUIRED_FEATURES.filter(
            feature =>
                !available.has(feature)
        );


    const labelColumn =
        LABEL_COLUMNS.find(
            column =>
                available.has(column)
        ) || null;


    return {

        compatible:
            missing.length === 0,

        missing,

        labelColumn

    };

}


/* =========================================================
   DISPLAY COMPATIBILITY
   ========================================================= */

function displayCompatibility(result) {

    const element =
        document.getElementById(
            "dataset-compatibility"
        );


    if (!element) {

        return;

    }


    if (result.compatible) {

        element.textContent =
            "✓ Dataset contains all required KDDCup99 features.";


        element.classList.remove(
            "incompatible"
        );


        element.classList.add(
            "compatible"
        );

    }

    else {

        element.textContent =
            "✓ CSV uploaded successfully. " +
            "Complete Dataset Analysis is available. " +
            "Single-record detection requires KDDCup99 fields.";


        element.classList.remove(
            "compatible"
        );


        element.classList.add(
            "incompatible"
        );

    }

}


/* =========================================================
   DATASET INFORMATION
   ========================================================= */

function updateDatasetInfo() {

    setText(
        "datasetName",
        selectedFile
            ? selectedFile.name
            : "—"
    );


    setText(
        "recordCount",
        csvRows.length.toLocaleString()
    );


    setText(
        "featureCount",
        csvHeaders.length
    );


    const label =
        LABEL_COLUMNS.find(
            column =>
                csvHeaders.includes(column)
        );


    const classes =
        label
            ? new Set(
                csvRows
                    .map(
                        row =>
                            String(
                                row[label] || ""
                            ).trim()
                    )
                    .filter(Boolean)
            )
            : new Set();


    setText(
        "classCount",
        classes.size || "—"
    );

}


/* =========================================================
   RESET DATASET INFORMATION
   ========================================================= */

function resetDatasetInfo() {

    setText(
        "datasetName",
        "—"
    );


    setText(
        "recordCount",
        "—"
    );


    setText(
        "featureCount",
        "—"
    );


    setText(
        "classCount",
        "—"
    );


    setText(
        "dataset-compatibility",
        ""
    );

}


/* =========================================================
   DETECT ATTACK
   ========================================================= */

async function detectAttack() {

    if (
        !selectedFile ||
        !csvRows.length
    ) {

        alert(
            "Please upload a dataset first."
        );

        return;

    }


    const compatibility =
        checkCompatibility();


    if (!compatibility.compatible) {

        alert(
            "Single-record detection requires " +
            "the 41 KDDCup99 features.\n\n" +
            "You can still use Complete Analysis " +
            "for this CSV."
        );

        return;

    }


    const recordNumberInput =
        document.getElementById(
            "record-number"
        );


    let recordNumber =
        Number(
            recordNumberInput
                ? recordNumberInput.value
                : 1
        );


    if (
        !Number.isInteger(recordNumber) ||
        recordNumber < 1 ||
        recordNumber > csvRows.length
    ) {

        recordNumber = 1;

    }


    const button =
        document.getElementById(
            "detect-button"
        );


    try {

        if (button) {

            button.disabled = true;

            button.textContent =
                "DETECTING...";

        }


        const record =
            prepareRecord(
                csvRows[
                    recordNumber - 1
                ]
            );


        const response =
            await fetchWithTimeout(
                API_URL + "/predict",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(record)

                },
                60000
            );


        const data =
            await readJSON(
                response
            );


        if (!response.ok) {

            throw new Error(
                extractBackendError(data)
            );

        }


        displayDetectionResult(
            data
        );


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        alert(
            "Prediction failed.\n\n" +
            error.message
        );


    } finally {

        if (button) {

            button.disabled = false;

            button.textContent =
                "DETECT ATTACK";

        }

    }

}


/* =========================================================
   PREPARE RECORD
   ========================================================= */

function prepareRecord(row) {

    const record = {};


    REQUIRED_FEATURES.forEach(
        feature => {

            if (
                CATEGORICAL_FEATURES
                    .includes(feature)
            ) {

                record[feature] =
                    row[feature] === undefined ||
                    row[feature] === ""
                        ? "Unknown"
                        : String(
                            row[feature]
                        );

            }

            else {

                const number =
                    Number(
                        row[feature]
                    );


                record[feature] =
                    Number.isFinite(number)
                        ? number
                        : 0;

            }

        }
    );


    return record;

}


/* =========================================================
   DISPLAY DETECTION RESULT
   ========================================================= */

function displayDetectionResult(data) {

    showElement(
        "detection-section"
    );


    const prediction =
        String(
            data.prediction ||
            "UNKNOWN"
        ).toUpperCase();


    const confidence =
        Number(
            data.confidence || 0
        );


    setText(
        "prediction",
        prediction
    );


    setText(
        "attackType",
        data.attack_type ||
        "Unknown"
    );


    setText(
        "confidence",
        confidence.toFixed(2) +
        "%"
    );


    setText(
        "detectionStatus",
        data.detection_status ||
        "Unknown"
    );


    setText(
        "iffAnomaly",
        data.iff_anomaly
            ? "TRUE"
            : "FALSE"
    );


    setText(
        "iffScore",
        Number(
            data.iff_score || 0
        ).toFixed(4)
    );


    const bar =
        document.getElementById(
            "confidence-bar"
        );


    if (bar) {

        bar.style.width =
            Math.max(
                0,
                Math.min(
                    100,
                    confidence
                )
            ) + "%";

    }


    const predictionElement =
        document.getElementById(
            "prediction"
        );


    if (predictionElement) {

        predictionElement.classList.remove(
            "attack",
            "normal"
        );


        predictionElement.classList.add(
            prediction === "ATTACK"
                ? "attack"
                : "normal"
        );

    }

}


/* =========================================================
   EVALUATE DATASET
   ========================================================= */

async function evaluateDataset() {

    console.log("========================================");
    console.log("STARTING DATASET EVALUATION");
    console.log("========================================");

    if (!selectedFile) {

        console.error(
            "No selected file."
        );

        alert(
            "Please upload a CSV dataset first."
        );

        return;
    }

    const button =
        document.getElementById(
            "evaluate-button"
        );

    try {

        if (button) {

            button.disabled = true;

            button.textContent =
                "EVALUATING...";
        }

        addLog(
            "EVALUATION",
            "Dataset evaluation started."
        );

        showElement(
            "evaluation-section"
        );

        setText(
            "accuracy",
            "Calculating..."
        );

        setText(
            "macroPrecision",
            "Calculating..."
        );

        setText(
            "macroRecall",
            "Calculating..."
        );

        setText(
            "macroF1",
            "Calculating..."
        );

        setText(
            "weightedF1",
            "Calculating..."
        );

        console.log(
            "Uploading:",
            selectedFile.name
        );

        const data =
            await uploadAnalysis();

        console.log(
            "EVALUATION RESPONSE:",
            data
        );

        displayEvaluation(data);

        addLog(
            "SUCCESS",
            "Dataset evaluation completed successfully."
        );

    } catch (error) {

        console.error(
            "Evaluation error:",
            error
        );

        addLog(
            "ERROR",
            "Model evaluation failed: " +
            error.message
        );

        alert(
            "Model evaluation failed.\n\n" +
            error.message
        );

    } finally {

        if (button) {

            button.disabled = false;

            button.textContent =
                "EVALUATE DATASET";
        }
    }
}

/* =========================================================
   COMPLETE ANALYSIS
   ========================================================= */

async function runCompleteAnalysis() {

    if (!selectedFile) {

        alert(
            "Please upload a CSV dataset first."
        );

        return;

    }


    const button =
        document.getElementById(
            "analysis-button"
        );


    try {

        if (button) {

            button.disabled = true;

            button.textContent =
                "RUNNING ANALYSIS...";

        }


        showElement(
            "analysis-section"
        );


        addLog(
            "ANALYSIS",
            "Complete analysis started."
        );


        /*
         * IMPORTANT:
         *
         * Send the actual selected CSV
         * directly to the backend.
         *
         * This prevents the frontend from
         * returning to the upload page.
         */

        const data =
            await uploadAnalysis();


        displayFullAnalysis(
            data
        );


        addLog(
            "COMPLETE",
            "Complete analysis finished successfully."
        );


        const section =
            document.getElementById(
                "analysis-section"
            );


        if (section) {

            section.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        }


    } catch (error) {

        console.error(
            "Complete analysis error:",
            error
        );


        addLog(
            "ERROR",
            "Complete analysis failed: " +
            error.message
        );


        alert(
            "Complete analysis failed.\n\n" +
            error.message
        );


    } finally {

        if (button) {

            button.disabled = false;

            button.textContent =
                "RUN COMPLETE ANALYSIS";

        }

    }

}


/* =========================================================
   UPLOAD DATASET TO BACKEND
   ========================================================= */

async function uploadAnalysis() {

    if (!selectedFile) {
        throw new Error(
            "No dataset selected."
        );
    }

    console.log(
        "Sending file to backend:",
        selectedFile.name
    );

    const formData =
        new FormData();

    formData.append(
        "file",
        selectedFile
    );

    const response =
        await fetchWithTimeout(
            API_URL +
            "/analyze-upload",
            {
                method: "POST",
                body: formData
            },
            600000
        );

    console.log(
        "Backend HTTP status:",
        response.status
    );

    const data =
        await readJSON(response);

    console.log(
        "Backend response:",
        data
    );

    if (!response.ok) {

        throw new Error(
            extractBackendError(data)
        );
    }

    if (
        data.success === false
    ) {

        throw new Error(
            data.message ||
            data.error ||
            "Backend analysis failed."
        );
    }

    return data;
}
/* =========================================================
   DISPLAY EVALUATION
   ========================================================= */

function displayEvaluation(data) {

    showElement(
        "evaluation-section"
    );


    const metrics =
        data.metrics ||
        data.overall_metrics ||
        {};


    setText(
        "accuracy",
        formatMetric(
            metrics.accuracy
        )
    );


    setText(
        "macroPrecision",
        formatMetric(
            metrics.macro_precision ??
            metrics.precision
        )
    );


    setText(
        "macroRecall",
        formatMetric(
            metrics.macro_recall ??
            metrics.recall
        )
    );


    setText(
        "macroF1",
        formatMetric(
            metrics.macro_f1 ??
            metrics.f1
        )
    );


    setText(
        "weightedF1",
        formatMetric(
            metrics.weighted_f1
        )
    );


    if (
        data.confusion_matrix
    ) {

        displayConfusionMatrix(
            data.confusion_matrix,
            "evaluation-confusion-matrix"
        );

    }

}


/* =========================================================
   DISPLAY COMPLETE ANALYSIS
   ========================================================= */

function displayFullAnalysis(data) {

    showElement(
        "analysis-section"
    );


    const dataset =
        data.dataset || {};


    setText(
        "datasetName",
        selectedFile
            ? selectedFile.name
            : "Uploaded Dataset"
    );


    setText(
        "recordCount",
        Number(
            dataset.original_records ??
            dataset.records_used ??
            csvRows.length
        ).toLocaleString()
    );


    setText(
        "featureCount",
        dataset.features ??
        csvHeaders.length
    );


    setText(
        "classCount",
        dataset.classes ??
        (
            Array.isArray(
                dataset.class_names
            )
                ? dataset.class_names.length
                : "—"
        )
    );


    displayModelComparison(
        data.comparison ||
        data.model_comparison ||
        []
    );


    if (
        data.class_performance
    ) {

        displayClassPerformance(
            data.class_performance
        );

    }

    else {

        deriveClassPerformance(
            data.confusion_matrix
        );

    }


    if (
        data.confusion_matrix
    ) {

        displayConfusionMatrix(
            data.confusion_matrix,
            "advanced-confusion-matrix"
        );

    }


    const rocData =
        data.roc ||
        data.benchmark?.roc ||
        data.analysis?.roc ||
        [];


    displayROC(
        rocData
    );

}


/* =========================================================
   MODEL COMPARISON
   ========================================================= */

function displayModelComparison(
    models
) {

    const body =
        document.getElementById(
            "comparison-body"
        );


    if (!body) {

        return;

    }


    body.innerHTML = "";


    if (
        !Array.isArray(models) ||
        !models.length
    ) {

        body.innerHTML =
            `
            <tr>
                <td colspan="6">
                    No model comparison data returned.
                </td>
            </tr>
            `;

        return;

    }


    models.forEach(
        model => {

            const row =
                document.createElement(
                    "tr"
                );


            const algorithm =
                model.algorithm ||
                model.name ||
                "Unknown";


            const precision =
                model.precision ??
                model.macro_precision;


            const recall =
                model.recall ??
                model.macro_recall;


            const f1 =
                model.f1 ??
                model.macro_f1;


            const time =
                model.training_time ??
                model.trainingTime;


            row.innerHTML =
                `
                <td>
                    ${escapeHTML(
                        algorithm
                    )}
                </td>

                <td>
                    ${formatMetric(
                        model.accuracy
                    )}
                </td>

                <td>
                    ${formatMetric(
                        precision
                    )}
                </td>

                <td>
                    ${formatMetric(
                        recall
                    )}
                </td>

                <td>
                    ${formatMetric(
                        f1
                    )}
                </td>

                <td>
                    ${
                        time !== undefined &&
                        time !== null
                            ? Number(time)
                                .toFixed(3) +
                              " s"
                            : "—"
                    }
                </td>
                `;


            body.appendChild(
                row
            );

        }
    );

}


/* =========================================================
   CLASS PERFORMANCE
   ========================================================= */

function displayClassPerformance(
    performance
) {

    const body =
        document.getElementById(
            "class-performance-body"
        );


    if (!body) {

        return;

    }


    body.innerHTML = "";


    if (
        !performance ||
        typeof performance !== "object" ||
        Array.isArray(performance)
    ) {

        body.innerHTML =
            `
            <tr>
                <td colspan="5">
                    No class performance available.
                </td>
            </tr>
            `;

        return;

    }


    Object.keys(
        performance
    ).forEach(
        className => {

            const item =
                performance[
                    className
                ] || {};


            const row =
                document.createElement(
                    "tr"
                );


            row.innerHTML =
                `
                <td>
                    ${escapeHTML(
                        className
                    )}
                </td>

                <td>
                    ${formatMetric(
                        item.precision
                    )}
                </td>

                <td>
                    ${formatMetric(
                        item.recall
                    )}
                </td>

                <td>
                    ${formatMetric(
                        item.f1
                    )}
                </td>

                <td>
                    ${
                        item.support !== undefined
                            ? Number(
                                item.support
                            ).toLocaleString()
                            : "—"
                    }
                </td>
                `;


            body.appendChild(
                row
            );

        }
    );


    if (
        !body.children.length
    ) {

        body.innerHTML =
            `
            <tr>
                <td colspan="5">
                    No class performance available.
                </td>
            </tr>
            `;

    }

}


/* =========================================================
   DERIVE CLASS PERFORMANCE
   ========================================================= */

function deriveClassPerformance(
    cm
) {

    if (
        !cm ||
        !Array.isArray(cm.matrix)
    ) {

        return;

    }


    const labels =
        Array.isArray(cm.labels)
            ? cm.labels
            : cm.matrix.map(
                (_, index) =>
                    "Class " + index
            );


    const matrix =
        cm.matrix;


    const performance = {};


    labels.forEach(
        (label, i) => {

            const tp =
                Number(
                    matrix[i]?.[i] || 0
                );


            let fp = 0;

            let fn = 0;

            let support = 0;


            for (
                let r = 0;
                r < matrix.length;
                r++
            ) {

                for (
                    let c = 0;
                    c <
                    (
                        matrix[r]?.length ||
                        0
                    );
                    c++
                ) {

                    const value =
                        Number(
                            matrix[r][c] ||
                            0
                        );


                    if (r === i) {

                        support += value;

                    }


                    if (
                        c === i &&
                        r !== i
                    ) {

                        fp += value;

                    }


                    if (
                        r === i &&
                        c !== i
                    ) {

                        fn += value;

                    }

                }

            }


            const precision =
                tp + fp
                    ? tp /
                      (tp + fp)
                    : 0;


            const recall =
                tp + fn
                    ? tp /
                      (tp + fn)
                    : 0;


            const f1 =
                precision + recall
                    ? (
                        2 *
                        precision *
                        recall
                    ) /
                    (
                        precision +
                        recall
                    )
                    : 0;


            performance[label] = {

                precision,

                recall,

                f1,

                support

            };

        }
    );


    displayClassPerformance(
        performance
    );

}


/* =========================================================
   CONFUSION MATRIX
   ========================================================= */

function displayConfusionMatrix(
    data,
    elementId
) {

    const container =
        document.getElementById(
            elementId
        );


    if (!container) {

        return;

    }


    const labels =
        Array.isArray(data?.labels)
            ? data.labels
            : [];


    const matrix =
        Array.isArray(data?.matrix)
            ? data.matrix
            : (
                Array.isArray(data)
                    ? data
                    : []
            );


    if (!matrix.length) {

        container.textContent =
            "Confusion matrix data unavailable.";

        return;

    }


    const finalLabels =
        labels.length
            ? labels
            : matrix.map(
                (_, index) =>
                    "Class " + index
            );


    let html =
        `
        <table>

            <thead>

                <tr>

                    <th>
                        Actual / Predicted
                    </th>
        `;


    finalLabels.forEach(
        label => {

            html +=
                `
                <th>
                    ${escapeHTML(label)}
                </th>
                `;

        }
    );


    html +=
        `
                </tr>

            </thead>

            <tbody>
        `;


    matrix.forEach(
        (row, r) => {

            html +=
                `
                <tr>

                    <th>
                        ${escapeHTML(
                            finalLabels[r] ||
                            "Class " + r
                        )}
                    </th>
                `;


            (
                Array.isArray(row)
                    ? row
                    : []
            ).forEach(
                value => {

                    html +=
                        `
                        <td>
                            ${Number(
                                value || 0
                            ).toLocaleString()}
                        </td>
                        `;

                }
            );


            html +=
                `
                </tr>
                `;

        }
    );


    html +=
        `
            </tbody>

        </table>
        `;


    container.innerHTML =
        html;

}


/* =========================================================
   ROC CURVE
   ========================================================= */

function displayROC(
    rocData
) {

    console.log(
        "ROC DATA RECEIVED:",
        rocData
    );


    const canvas =
        document.getElementById(
            "roc-chart"
        );


    const message =
        document.getElementById(
            "roc-message"
        );


    if (!canvas) {

        return;

    }


    if (
        typeof Chart ===
        "undefined"
    ) {

        console.warn(
            "Chart.js is not loaded."
        );

        if (message) {

            message.textContent =
                "Chart.js could not be loaded.";

        }

        return;

    }


    let curves = [];


    if (
        Array.isArray(rocData)
    ) {

        curves = rocData;

    }

    else if (
        rocData &&
        Array.isArray(
            rocData.curves
        )
    ) {

        curves =
            rocData.curves;

    }

    else if (
        rocData &&
        typeof rocData ===
            "object"
    ) {

        Object.keys(
            rocData
        ).forEach(
            key => {

                const item =
                    rocData[key];


                if (
                    item &&
                    Array.isArray(
                        item.fpr
                    ) &&
                    Array.isArray(
                        item.tpr
                    )
                ) {

                    curves.push({

                        class_name:
                            item.class_name ||
                            key,

                        fpr:
                            item.fpr,

                        tpr:
                            item.tpr,

                        auc:
                            item.auc

                    });

                }

            }
        );

    }


    if (!curves.length) {

        console.warn(
            "No ROC data returned."
        );


        if (message) {

            message.textContent =
                "ROC curve data is not available for this analysis.";

        }


        return;

    }


    if (message) {

        message.textContent = "";

    }


    if (rocChart) {

        rocChart.destroy();

        rocChart = null;

    }


    const datasets = [];


    curves.forEach(
        (curve, index) => {

            if (
                !curve ||
                !Array.isArray(
                    curve.fpr
                ) ||
                !Array.isArray(
                    curve.tpr
                )
            ) {

                return;

            }


            const points = [];


            const length =
                Math.min(
                    curve.fpr.length,
                    curve.tpr.length
                );


            for (
                let i = 0;
                i < length;
                i++
            ) {

                points.push({

                    x:
                        Number(
                            curve.fpr[i]
                        ),

                    y:
                        Number(
                            curve.tpr[i]
                        )

                });

            }


            datasets.push({

                label:
                    (
                        curve.class_name ||
                        curve.className ||
                        "Class " +
                        (index + 1)
                    ) +

                    (
                        curve.auc !== undefined
                            ? " (AUC: " +
                              Number(
                                  curve.auc
                              ).toFixed(3) +
                              ")"
                            : ""
                    ),

                data: points,

                parsing: false,

                fill: false,

                tension: 0.15,

                borderWidth: 2,

                pointRadius: 0

            });

        }
    );


    datasets.push({

        label:
            "Random Classifier",

        data: [
            {
                x: 0,
                y: 0
            },
            {
                x: 1,
                y: 1
            }
        ],

        parsing: false,

        fill: false,

        borderWidth: 1,

        borderDash: [
            6,
            6
        ],

        pointRadius: 0

    });


    rocChart =
        new Chart(
            canvas,
            {

                type: "line",

                data: {
                    datasets
                },

                options: {

                    responsive: true,

                    maintainAspectRatio:
                        false,

                    scales: {

                        x: {

                            type: "linear",

                            min: 0,

                            max: 1,

                            title: {
                                display: true,

                                text:
                                    "False Positive Rate"
                            }

                        },

                        y: {

                            min: 0,

                            max: 1,

                            title: {
                                display: true,

                                text:
                                    "True Positive Rate"
                            }

                        }

                    },

                    plugins: {

                        legend: {
                            display: true
                        },

                        title: {

                            display: true,

                            text:
                                "ROC Curve"

                        }

                    }

                }

            }
        );

}


/* =========================================================
   BUTTON HELPERS
   ========================================================= */

function disableButtons() {

    [
        "detect-button",
        "evaluate-button",
        "analysis-button"
    ].forEach(
        id => {

            const button =
                document.getElementById(
                    id
                );


            if (button) {

                button.disabled = true;

            }

        }
    );

}


function enableAnalysisButtons() {

    [
        "evaluate-button",
        "analysis-button"
    ].forEach(
        id => {

            const button =
                document.getElementById(
                    id
                );


            if (button) {

                button.disabled = false;

            }

        }
    );

}


/* =========================================================
   UI HELPERS
   ========================================================= */

function showElement(id) {

    const element =
        document.getElementById(id);


    if (element) {

        element.classList.remove(
            "hidden"
        );

    }

}


function hideElement(id) {

    const element =
        document.getElementById(id);


    if (element) {

        element.classList.add(
            "hidden"
        );

    }

}


function setText(
    id,
    value
) {

    const element =
        document.getElementById(id);


    if (element) {

        element.textContent =
            value === undefined ||
            value === null
                ? "—"
                : String(value);

    }

}


/* =========================================================
   METRIC FORMAT
   ========================================================= */

function formatMetric(value) {

    if (
        value === undefined ||
        value === null ||
        value === ""
    ) {

        return "—";

    }


    const number =
        Number(value);


    if (
        !Number.isFinite(number)
    ) {

        return String(value);

    }


    return (
        number <= 1
            ? number * 100
            : number
    ).toFixed(2) + "%";

}


/* =========================================================
   HTML ESCAPE
   ========================================================= */

function escapeHTML(value) {

    return String(
        value ?? ""
    )

        .replace(
            /&/g,
            "&amp;"
        )

        .replace(
            /</g,
            "&lt;"
        )

        .replace(
            />/g,
            "&gt;"
        )

        .replace(
            /"/g,
            "&quot;"
        )

        .replace(
            /'/g,
            "&#039;"
        );

}


/* =========================================================
   ACTIVITY LOG
   ========================================================= */

function addLog(
    type,
    message
) {

    console.log(
        "[" +
        type +
        "]",
        message
    );


    const log =
        document.getElementById(
            "activity-log"
        );


    if (!log) {

        return;

    }


    const line =
        document.createElement(
            "div"
        );


    line.className =
        "log-line";


    line.textContent =
        type +
        ": " +
        message;


    log.prepend(
        line
    );

}


/* =========================================================
   BACKEND ERROR
   ========================================================= */

function extractBackendError(
    data
) {

    if (!data) {

        return "Unknown backend error.";

    }


    if (
        typeof data.detail ===
        "string"
    ) {

        return data.detail;

    }


    if (
        data.detail &&
        typeof data.detail ===
            "object"
    ) {

        return (
            data.detail.message ||
            data.detail.error ||
            JSON.stringify(
                data.detail
            )
        );

    }


    return (
        data.message ||
        data.error ||
        "Backend request failed."
    );

}


/* =========================================================
   GLOBAL ERROR LOGGING
   ========================================================= */

window.addEventListener(
    "error",
    event => {

        console.error(
            "Frontend error:",
            event.error ||
            event.message
        );

    }
);


window.addEventListener(
    "unhandledrejection",
    event => {

        console.error(
            "Unhandled promise rejection:",
            event.reason
        );

    }
);


console.log(
    "Universal Cyber Detection app.js loaded."
);
console.log(
    "🔥🔥🔥 NEW APP.JS IS RUNNING 🔥🔥🔥"
);

console.log(
    "APP JS VERSION: 2026-09-26-FIXED"
);
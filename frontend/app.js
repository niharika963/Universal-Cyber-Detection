"use strict";

/*
============================================================
 UNIVERSAL CYBER ATTACK DETECTION SYSTEM
 IALP + IFF + XGBoost
============================================================
 Frontend:
 1. Upload CSV
 2. Read CSV
 3. Check compatibility
 4. Display dataset information
 5. Detect attack
 6. Evaluate trained model
 7. Run complete analysis
============================================================
*/

const API_URL = "http://127.0.0.1:8000";

let selectedFile = null;
let csvRows = [];
let csvHeaders = [];
let rocChart = null;


/* ============================================================
   REQUIRED KDDCup99 FEATURES
============================================================ */

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


const POSSIBLE_LABEL_COLUMNS = [
    "label",
    "class",
    "attack",
    "attack_type",
    "connection_type",
    "target",
    "category"
];


/* ============================================================
   DOM READY
============================================================ */

document.addEventListener("DOMContentLoaded", function () {

    console.log("========================================");
    console.log("Cyber Detection Frontend Loaded");
    console.log("========================================");

    const fileInput =
        document.getElementById("file-input");

    const detectButton =
        document.getElementById("detect-button");

    const evaluateButton =
        document.getElementById("evaluate-button");

    const analysisButton =
        document.getElementById("analysis-button");


    if (fileInput) {
        fileInput.addEventListener(
            "change",
            handleFileSelection
        );
    }


    if (detectButton) {
        detectButton.addEventListener(
            "click",
            detectAttack
        );
    }


    if (evaluateButton) {
        evaluateButton.addEventListener(
            "click",
            evaluateDataset
        );
    }


    if (analysisButton) {
        analysisButton.addEventListener(
            "click",
            runCompleteAnalysis
        );
    }


    disableActionButtons();

    checkBackend();

});


/* ============================================================
   SHOW ELEMENT
============================================================ */

function showElement(id) {

    const element =
        document.getElementById(id);

    if (element) {
        element.classList.remove("hidden");

        /*
        Some CSS implementations use display:none
        instead of the hidden class.
        */
        element.style.display = "";
    }
}


/* ============================================================
   HIDE ELEMENT
============================================================ */

function hideElement(id) {

    const element =
        document.getElementById(id);

    if (element) {
        element.classList.add("hidden");
    }
}


/* ============================================================
   DISABLE ACTION BUTTONS
============================================================ */

function disableActionButtons() {

    const detectButton =
        document.getElementById("detect-button");

    const evaluateButton =
        document.getElementById("evaluate-button");

    const analysisButton =
        document.getElementById("analysis-button");


    if (detectButton) {
        detectButton.disabled = true;
    }

    if (evaluateButton) {
        evaluateButton.disabled = true;
    }

    if (analysisButton) {
        analysisButton.disabled = true;
    }
}


/* ============================================================
   ENABLE DATASET ACTIONS
============================================================ */

function enableDatasetActions() {

    const detectButton =
        document.getElementById("detect-button");

    const evaluateButton =
        document.getElementById("evaluate-button");

    const analysisButton =
        document.getElementById("analysis-button");


    if (detectButton) {
        detectButton.disabled = false;
    }

    if (evaluateButton) {
        evaluateButton.disabled = false;
    }

    if (analysisButton) {
        analysisButton.disabled = false;
    }


    /*
    IMPORTANT FIX:

    These sections were previously hidden after
    dataset upload and only the buttons were enabled.

    Now the sections themselves are displayed.
    */

    showElement("detection-section");
    showElement("evaluation-section");
    showElement("analysis-section");
}


/* ============================================================
   BACKEND HEALTH CHECK
============================================================ */

async function checkBackend() {

    const status =
        document.getElementById("backend-status");

    const indicator =
        document.getElementById("backend-indicator");


    try {

        const response =
            await fetch(API_URL + "/health");


        if (!response.ok) {
            throw new Error(
                "Backend health check failed."
            );
        }


        const data =
            await response.json();


        console.log(
            "Backend health:",
            data
        );


        if (status) {
            status.textContent =
                "Backend Connected";
        }


        if (indicator) {
            indicator.style.background =
                "#35d07f";
        }

    }
    catch (error) {

        console.error(
            "Backend connection error:",
            error
        );


        if (status) {
            status.textContent =
                "Backend Offline";
        }


        if (indicator) {
            indicator.style.background =
                "#ff5f5f";
        }

    }
}


/* ============================================================
   FILE SELECTION
============================================================ */

async function handleFileSelection(event) {

    const input =
        event.target;


    selectedFile =
        input.files[0];


    csvRows = [];
    csvHeaders = [];


    disableActionButtons();


    /*
    Hide the sections only while loading.

    After a compatible dataset is loaded,
    enableDatasetActions() will show them again.
    */

    hideElement("detection-section");
    hideElement("evaluation-section");
    hideElement("analysis-section");


    const fileStatus =
        document.getElementById("file-status");


    if (!selectedFile) {

        setText(
            "file-status",
            "No dataset selected"
        );

        resetDatasetInformation();

        return;
    }


    if (
        !selectedFile.name
            .toLowerCase()
            .endsWith(".csv")
    ) {

        alert(
            "Please select a CSV dataset."
        );

        resetFileInput();

        return;
    }


    if (fileStatus) {

        fileStatus.textContent =
            "Reading " +
            selectedFile.name +
            "...";
    }


    try {

        await readCSV();


        if (csvRows.length === 0) {

            throw new Error(
                "No valid CSV records were found."
            );
        }


        updateLocalDatasetInformation();


        const compatibility =
            checkDatasetCompatibility();


        displayCompatibility(
            compatibility
        );


        if (compatibility.compatible) {

            if (fileStatus) {

                fileStatus.textContent =
                    selectedFile.name +
                    " loaded successfully — " +
                    csvRows.length.toLocaleString() +
                    " records";
            }


            enableDatasetActions();

        }
        else {

            if (fileStatus) {

                fileStatus.textContent =
                    selectedFile.name +
                    " loaded — dataset adapter required";
            }

        }

    }
    catch (error) {

        console.error(
            "CSV loading error:",
            error
        );


        setText(
            "file-status",
            "Failed to load dataset"
        );


        resetDatasetInformation();


        alert(
            "Dataset loading failed.\n\n" +
            error.message
        );

    }
}


/* ============================================================
   RESET FILE INPUT
============================================================ */

function resetFileInput() {

    const input =
        document.getElementById("file-input");


    if (input) {
        input.value = "";
    }


    selectedFile = null;
    csvRows = [];
    csvHeaders = [];


    disableActionButtons();


    hideElement("detection-section");
    hideElement("evaluation-section");
    hideElement("analysis-section");
}


/* ============================================================
   RESET DATASET INFORMATION
============================================================ */

function resetDatasetInformation() {

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


    const compatibility =
        document.getElementById(
            "dataset-compatibility"
        );


    if (compatibility) {

        compatibility.textContent =
            "Dataset compatibility will be checked after upload.";

        compatibility.style.color = "";
    }
}


/* ============================================================
   READ CSV
============================================================ */

async function readCSV() {

    const text =
        await selectedFile.text();


    const lines =
        text
            .split(/\r?\n/)
            .filter(
                line =>
                    line.trim() !== ""
            );


    if (lines.length < 2) {

        throw new Error(
            "CSV file must contain a header and at least one data row."
        );
    }


    const headers =
        parseCSVLine(lines[0])
            .map(
                header =>
                    cleanHeader(header)
            );


    if (headers.length === 0) {

        throw new Error(
            "CSV header could not be read."
        );
    }


    csvHeaders = headers;
    csvRows = [];


    for (
        let i = 1;
        i < lines.length;
        i++
    ) {

        const values =
            parseCSVLine(lines[i]);


        if (
            values.length !==
            headers.length
        ) {

            console.warn(
                "Skipping malformed CSV row:",
                i + 1
            );

            continue;
        }


        const row = {};


        headers.forEach(
            function (header, index) {

                row[header] =
                    cleanCSVValue(
                        values[index]
                    );

            }
        );


        csvRows.push(row);
    }


    console.log(
        "CSV headers:",
        csvHeaders
    );


    console.log(
        "CSV rows:",
        csvRows.length
    );
}


/* ============================================================
   CLEAN HEADER
============================================================ */

function cleanHeader(header) {

    return String(header)
        .trim()
        .replace(/^"|"$/g, "")
        .trim();
}


/* ============================================================
   CLEAN CSV VALUE
============================================================ */

function cleanCSVValue(value) {

    return String(value ?? "")
        .trim()
        .replace(/^"|"$/g, "");
}


/* ============================================================
   CSV PARSER
============================================================ */

function parseCSVLine(line) {

    const result = [];

    let current = "";

    let insideQuotes = false;


    for (
        let i = 0;
        i < line.length;
        i++
    ) {

        const char =
            line[i];


        if (char === '"') {

            if (
                insideQuotes &&
                line[i + 1] === '"'
            ) {

                current += '"';

                i++;

            }
            else {

                insideQuotes =
                    !insideQuotes;
            }

        }
        else if (
            char === "," &&
            !insideQuotes
        ) {

            result.push(current);

            current = "";

        }
        else {

            current += char;
        }
    }


    result.push(current);


    return result;
}


/* ============================================================
   UPDATE DATASET INFORMATION
============================================================ */

function updateLocalDatasetInformation() {

    const featureCount =
        csvHeaders.length;


    const labelColumn =
        findLabelColumn(
            csvHeaders
        );


    const actualFeatureCount =
        labelColumn
            ? featureCount - 1
            : featureCount;


    setText(
        "datasetName",
        selectedFile
            ? selectedFile.name
            : "Unknown"
    );


    setText(
        "recordCount",
        csvRows.length.toLocaleString()
    );


    setText(
        "featureCount",
        actualFeatureCount
    );


    const classes =
        getDetectedClasses(
            labelColumn
        );


    setText(
        "classCount",
        classes.length > 0
            ? classes.length
            : "—"
    );


    console.log(
        "Dataset information:",
        {
            name:
                selectedFile
                    ? selectedFile.name
                    : null,

            rows:
                csvRows.length,

            columns:
                csvHeaders.length,

            features:
                actualFeatureCount,

            label:
                labelColumn,

            classes:
                classes
        }
    );
}


/* ============================================================
   FIND LABEL COLUMN
============================================================ */

function findLabelColumn(headers) {

    const normalized =
        headers.map(
            header =>
                String(header)
                    .trim()
                    .toLowerCase()
        );


    for (
        const possible
        of POSSIBLE_LABEL_COLUMNS
    ) {

        const index =
            normalized.indexOf(
                possible
            );


        if (index !== -1) {

            return headers[index];
        }
    }


    return null;
}


/* ============================================================
   GET CLASSES
============================================================ */

function getDetectedClasses(labelColumn) {

    if (
        !labelColumn ||
        csvRows.length === 0
    ) {

        return [];
    }


    const classSet =
        new Set();


    csvRows.forEach(
        function (row) {

            const value =
                String(
                    row[labelColumn] ?? ""
                ).trim();


            if (value !== "") {

                classSet.add(value);
            }
        }
    );


    return Array.from(classSet);
}


/* ============================================================
   CHECK DATASET COMPATIBILITY
============================================================ */

function checkDatasetCompatibility() {

    const normalizedHeaders =
        csvHeaders.map(
            header =>
                header
                    .trim()
                    .toLowerCase()
        );


    const missingFeatures =
        REQUIRED_FEATURES.filter(
            feature =>
                !normalizedHeaders.includes(
                    feature.toLowerCase()
                )
        );


    return {

        compatible:
            missingFeatures.length === 0,

        missingFeatures:
            missingFeatures,

        featureCount:
            csvHeaders.length,

        rows:
            csvRows.length,

        labelColumn:
            findLabelColumn(
                csvHeaders
            )

    };
}


/* ============================================================
   DISPLAY COMPATIBILITY
============================================================ */

function displayCompatibility(
    compatibility
) {

    const element =
        document.getElementById(
            "dataset-compatibility"
        );


    if (!element) {
        return;
    }


    if (compatibility.compatible) {

        element.textContent =
            "✓ Dataset is compatible with the current detection model.";

        element.style.color =
            "#35d07f";

        return;
    }


    element.textContent =
        "Dataset detected. This dataset format is not yet supported by the current model.";


    element.style.color =
        "#f0b35b";


    console.log(
        "Missing features:",
        compatibility.missingFeatures
    );
}


/* ============================================================
   PREPARE RECORD
============================================================ */

function prepareRecord(row) {

    const record = {};


    REQUIRED_FEATURES.forEach(
        function (feature) {

            const value =
                row[feature];


            if (
                value === undefined ||
                value === null
            ) {

                return;
            }


            const stringValue =
                String(value).trim();


            if (
                CATEGORICAL_FEATURES.includes(
                    feature
                )
            ) {

                record[feature] =
                    stringValue;

                return;
            }


            const number =
                Number(stringValue);


            if (
                stringValue !== "" &&
                Number.isFinite(number)
            ) {

                record[feature] =
                    number;

            }
            else {

                record[feature] =
                    stringValue;
            }
        }
    );


    return record;
}


/* ============================================================
   DETECT ATTACK
============================================================ */

async function detectAttack() {

    const button =
        document.getElementById(
            "detect-button"
        );


    if (csvRows.length === 0) {

        alert(
            "Please upload a cybersecurity dataset first."
        );

        return;
    }


    const compatibility =
        checkDatasetCompatibility();


    if (!compatibility.compatible) {

        alert(
            "This dataset is not compatible with the current trained model."
        );

        return;
    }


    const numberInput =
        document.getElementById(
            "record-number"
        );


    let recordNumber =
        Number(
            numberInput
                ? numberInput.value
                : 1
        );


    if (
        !Number.isInteger(recordNumber) ||
        recordNumber < 1
    ) {

        recordNumber = 1;
    }


    if (recordNumber > csvRows.length) {

        alert(
            "Record number must be between 1 and " +
            csvRows.length.toLocaleString()
        );

        return;
    }


    const row =
        csvRows[
            recordNumber - 1
        ];


    const record =
        prepareRecord(row);


    console.log(
        "Sending record:",
        record
    );


    try {

        if (button) {

            button.disabled = true;

            button.textContent =
                "DETECTING...";
        }


        const response =
            await fetch(
                API_URL + "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(record)
                }
            );


        const data =
            await response.json();


        console.log(
            "Prediction response:",
            data
        );


        if (!response.ok) {

            throw new Error(
                extractBackendError(data)
            );
        }


        displayDetectionResult(data);

    }
    catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        alert(
            "Prediction failed.\n\n" +
            error.message
        );

    }
    finally {

        if (button) {

            button.disabled = false;

            button.textContent =
                "DETECT ATTACK";
        }
    }
}


/* ============================================================
   DISPLAY DETECTION RESULT
============================================================ */

function displayDetectionResult(data) {

    showElement("detection-section");


    const prediction =
        String(
            data.prediction ??
            "UNKNOWN"
        );


    const attackType =
        String(
            data.attack_type ??
            "Unknown"
        );


    const confidence =
        Number(
            data.confidence ?? 0
        );


    const detectionStatus =
        String(
            data.detection_status ??
            (
                prediction.toUpperCase() === "ATTACK"
                    ? "Malicious Traffic"
                    : "Benign Traffic"
            )
        );


    const iffAnomaly =
        data.iff_anomaly;


    const iffScore =
        data.iff_score;


    setText(
        "prediction",
        prediction
    );


    setText(
        "attackType",
        attackType
    );


    setText(
        "confidence",
        confidence.toFixed(2) + "%"
    );


    setText(
        "confidence-value",
        confidence.toFixed(2) + "%"
    );


    const confidenceBar =
        document.getElementById(
            "confidence-bar"
        );


    if (confidenceBar) {

        const safeConfidence =
            Math.max(
                0,
                Math.min(
                    100,
                    confidence
                )
            );


        confidenceBar.style.width =
            safeConfidence + "%";
    }


    setText(
        "detectionStatus",
        detectionStatus
    );


    setText(
        "iffAnomaly",
        iffAnomaly === true
            ? "True"
            : iffAnomaly === false
                ? "False"
                : "—"
    );


    setText(
        "iffScore",
        Number.isFinite(
            Number(iffScore)
        )
            ? Number(iffScore).toFixed(4)
            : "—"
    );


    /*
    Optional visual classes.
    */

    const predictionElement =
        document.getElementById(
            "prediction"
        );


    if (predictionElement) {

        predictionElement.classList.remove(
            "attack",
            "normal",
            "malicious",
            "benign"
        );


        if (
            prediction.toUpperCase() ===
            "ATTACK"
        ) {

            predictionElement.classList.add(
                "attack"
            );

        }
        else {

            predictionElement.classList.add(
                "normal"
            );
        }
    }
}


/* ============================================================
   EVALUATE DATASET
============================================================ */

async function evaluateDataset() {

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


        const response =
            await fetch(
                API_URL + "/analyze"
            );


        const data =
            await response.json();


        console.log(
            "Evaluation response:",
            data
        );


        if (!response.ok) {

            throw new Error(
                extractBackendError(data)
            );
        }


        displayEvaluation(data);

    }
    catch (error) {

        console.error(
            "Evaluation error:",
            error
        );


        alert(
            "Dataset evaluation failed.\n\n" +
            error.message
        );

    }
    finally {

        if (button) {

            button.disabled = false;

            button.textContent =
                "EVALUATE DATASET";
        }
    }
}


/* ============================================================
   DISPLAY EVALUATION
============================================================ */

function displayEvaluation(data) {

    showElement("evaluation-section");


    const model =
        data.proposed_model ||
        data.model ||
        {};


    const metrics =
        model.metrics ||
        data.metrics ||
        {};


    setText(
        "accuracy",
        metricValue(
            metrics.accuracy
        )
    );


    setText(
        "macroPrecision",
        metricValue(
            metrics.macro_precision ??
            metrics.macroPrecision ??
            metrics.precision
        )
    );


    setText(
        "macroRecall",
        metricValue(
            metrics.macro_recall ??
            metrics.macroRecall ??
            metrics.recall
        )
    );


    setText(
        "macroF1",
        metricValue(
            metrics.macro_f1 ??
            metrics.macroF1 ??
            metrics.f1
        )
    );


    setText(
        "weightedF1",
        metricValue(
            metrics.weighted_f1 ??
            metrics.weightedF1
        )
    );


    const matrix =
        model.confusion_matrix ||
        data.confusion_matrix;


    if (matrix) {

        displayConfusionMatrix(
            matrix,
            "evaluation-confusion-matrix"
        );
    }
}


/* ============================================================
   COMPLETE ANALYSIS
============================================================ */

async function runCompleteAnalysis() {

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


        const response =
            await fetch(
                API_URL + "/analyze/refresh",
                {
                    method: "POST"
                }
            );


        const data =
            await response.json();


        console.log(
            "Complete analysis response:",
            data
        );


        if (!response.ok) {

            throw new Error(
                extractBackendError(data)
            );
        }


        displayFullAnalysis(data);

    }
    catch (error) {

        console.error(
            "Complete analysis error:",
            error
        );


        alert(
            "Complete analysis failed.\n\n" +
            error.message
        );

    }
    finally {

        if (button) {

            button.disabled = false;

            button.textContent =
                "RUN COMPLETE ANALYSIS";
        }
    }
}


/* ============================================================
   DISPLAY FULL ANALYSIS
============================================================ */

function displayFullAnalysis(data) {

    showElement("analysis-section");


    /*
    Display evaluation metrics too.
    */

    displayEvaluation(data);


    /*
    Model comparison
    */

    displayComparison(
        data.comparison ||
        data.algorithm_comparison ||
        []
    );


    /*
    Class performance
    */

    displayClassPerformance(
        data.class_performance ||
        data.class_metrics ||
        {}
    );


    /*
    Confusion matrix
    */

    const matrix =
        data.confusion_matrix ||
        data.proposed_model?.confusion_matrix;


    if (matrix) {

        displayConfusionMatrix(
            matrix,
            "advanced-confusion-matrix"
        );
    }


    /*
    ROC
    */

    if (data.roc) {

        displayROC(
            data.roc
        );
    }


    showElement("advanced-section");
    showElement("class-section");


    const section =
        document.getElementById(
            "analysis-section"
        );


    if (section) {

        section.scrollIntoView({
            behavior: "smooth"
        });
    }
}


/* ============================================================
   MODEL COMPARISON
============================================================ */

function displayComparison(comparison) {

    const body =
        document.getElementById(
            "comparison-body"
        );


    if (!body) {
        return;
    }


    body.innerHTML = "";


    if (
        !Array.isArray(comparison) ||
        comparison.length === 0
    ) {

        body.innerHTML = `
            <tr>
                <td colspan="5">
                    No model comparison data available.
                </td>
            </tr>
        `;

        return;
    }


    comparison.forEach(
        function (model) {

            const row =
                document.createElement("tr");


            row.innerHTML = `
                <td>
                    ${escapeHTML(
                        model.algorithm ||
                        model.name ||
                        "Model"
                    )}
                </td>

                <td>
                    ${metricValue(
                        model.accuracy
                    )}
                </td>

                <td>
                    ${metricValue(
                        model.precision
                    )}
                </td>

                <td>
                    ${metricValue(
                        model.recall
                    )}
                </td>

                <td>
                    ${metricValue(
                        model.f1
                    )}
                </td>
            `;


            body.appendChild(row);
        }
    );
}


/* ============================================================
   CLASS PERFORMANCE
============================================================ */

function displayClassPerformance(performance) {

    const body =
        document.getElementById(
            "class-performance-body"
        );


    if (!body) {
        return;
    }


    body.innerHTML = "";


    if (Array.isArray(performance)) {

        performance.forEach(
            function (item) {

                addClassRow(
                    body,
                    item.class ||
                    item.label ||
                    "Class",
                    item
                );
            }
        );

        return;
    }


    if (
        performance &&
        typeof performance === "object"
    ) {

        Object.keys(performance)
            .forEach(
                function (className) {

                    addClassRow(
                        body,
                        className,
                        performance[className]
                    );
                }
            );
    }


    if (body.children.length === 0) {

        body.innerHTML = `
            <tr>
                <td colspan="5">
                    No class performance data available.
                </td>
            </tr>
        `;
    }
}


/* ============================================================
   ADD CLASS ROW
============================================================ */

function addClassRow(
    body,
    className,
    item
) {

    item = item || {};


    const row =
        document.createElement("tr");


    row.innerHTML = `
        <td>
            ${escapeHTML(className)}
        </td>

        <td>
            ${metricValue(item.precision)}
        </td>

        <td>
            ${metricValue(item.recall)}
        </td>

        <td>
            ${metricValue(item.f1)}
        </td>

        <td>
            ${Number(
                item.support || 0
            ).toLocaleString()}
        </td>
    `;


    body.appendChild(row);
}


/* ============================================================
   CONFUSION MATRIX
============================================================ */

function displayConfusionMatrix(
    matrixData,
    elementId
) {

    const container =
        document.getElementById(
            elementId
        );


    if (!container) {
        return;
    }


    let labels;
    let matrix;


    if (
        matrixData &&
        matrixData.labels &&
        matrixData.matrix
    ) {

        labels =
            matrixData.labels;

        matrix =
            matrixData.matrix;

    }
    else {

        labels = [
            "Normal",
            "DoS",
            "Probe",
            "R2L",
            "U2R"
        ];

        matrix =
            matrixData;
    }


    if (!Array.isArray(matrix)) {

        container.textContent =
            "Confusion matrix data unavailable.";

        return;
    }


    let html =
        "<table class='confusion-table'>";


    html +=
        "<thead><tr>";


    html +=
        "<th>Actual / Predicted</th>";


    labels.forEach(
        function (label) {

            html +=
                "<th>" +
                escapeHTML(label) +
                "</th>";
        }
    );


    html +=
        "</tr></thead>";


    html +=
        "<tbody>";


    matrix.forEach(
        function (row, index) {

            html +=
                "<tr>";


            html +=
                "<th>" +
                escapeHTML(
                    labels[index] ||
                    "Class"
                ) +
                "</th>";


            if (Array.isArray(row)) {

                row.forEach(
                    function (value) {

                        html +=
                            "<td>" +
                            Number(
                                value || 0
                            ).toLocaleString() +
                            "</td>";
                    }
                );
            }


            html +=
                "</tr>";
        }
    );


    html +=
        "</tbody></table>";


    container.innerHTML =
        html;
}


/* ============================================================
   ROC CURVE
============================================================ */

function displayROC(rocData) {

    const canvas =
        document.getElementById(
            "roc-chart"
        );


    if (!canvas) {
        return;
    }


    if (
        typeof Chart ===
        "undefined"
    ) {

        console.warn(
            "Chart.js is not available."
        );

        return;
    }


    if (
        !rocData ||
        typeof rocData !== "object"
    ) {

        return;
    }


    const datasets = [];


    Object.keys(rocData)
        .forEach(
            function (className) {

                const item =
                    rocData[className];


                if (
                    !item ||
                    !Array.isArray(item.fpr) ||
                    !Array.isArray(item.tpr)
                ) {

                    return;
                }


                const points =
                    item.fpr.map(
                        function (fpr, index) {

                            return {
                                x: Number(fpr),
                                y: Number(
                                    item.tpr[index]
                                )
                            };
                        }
                    );


                datasets.push({

                    label:
                        className +
                        (
                            item.auc !== undefined
                                ? " (AUC " +
                                  Number(
                                      item.auc
                                  ).toFixed(4) +
                                  ")"
                                : ""
                        ),

                    data:
                        points,

                    fill:
                        false,

                    tension:
                        0.15
                });
            }
        );


    if (datasets.length === 0) {
        return;
    }


    if (rocChart) {
        rocChart.destroy();
    }


    rocChart =
        new Chart(
            canvas,
            {
                type: "line",

                data: {
                    datasets: datasets
                },

                options: {

                    responsive: true,

                    maintainAspectRatio:
                        false,

                    parsing: false,

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
                    }
                }
            }
        );
}


/* ============================================================
   BACKEND ERROR
============================================================ */

function extractBackendError(data) {

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

        if (data.detail.message) {

            let message =
                data.detail.message;


            if (
                Array.isArray(
                    data.detail.missing_features
                )
            ) {

                message +=
                    "\n\nMissing features:\n" +
                    data.detail.missing_features.join(
                        ", "
                    );
            }


            return message;
        }


        return JSON.stringify(
            data.detail
        );
    }


    if (data.error) {

        return String(
            data.error
        );
    }


    return "Backend request failed.";
}


/* ============================================================
   SET TEXT
============================================================ */

function setText(id, value) {

    const element =
        document.getElementById(id);


    if (element) {

        element.textContent =
            value;
    }
}


/* ============================================================
   METRIC FORMAT
============================================================ */

function metricValue(value) {

    if (
        value === undefined ||
        value === null ||
        value === ""
    ) {

        return "—";
    }


    const number =
        Number(value);


    if (!Number.isFinite(number)) {

        return "—";
    }


    /*
    Backend metrics may be returned as:

    0.9995
    OR
    99.95

    Convert decimal metrics to percentage.
    */

    const percentage =
        number <= 1
            ? number * 100
            : number;


    return percentage.toFixed(2) + "%";
}


/* ============================================================
   ESCAPE HTML
============================================================ */

function escapeHTML(value) {

    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


/* ============================================================
   DEBUG HELPER
============================================================ */

console.log(
    "Universal Cyber Detection app.js loaded successfully."
);
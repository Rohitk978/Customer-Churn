
const form = document.getElementById("predictionForm");
const predictButton = document.getElementById("predictButton");
const resetButton = document.getElementById("resetButton");

const resultCard = document.getElementById("resultCard");
const resultTitle = document.getElementById("resultTitle");
const resultBadge = document.getElementById("resultBadge");
const probabilityValue = document.getElementById("probabilityValue");
const progressBar = document.getElementById("progressBar");
const resultMessage = document.getElementById("resultMessage");

const errorCard = document.getElementById("errorCard");
const errorMessage = document.getElementById("errorMessage");

const apiStatus = document.getElementById("apiStatus");
const statusText = document.getElementById("statusText");


/* =========================================================
   ALL FORM FIELDS
   ========================================================= */

const fields = [
    "gender",
    "partner",
    "dependents",

    "phoneservice",
    "multiplelines",

    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",

    "streamingtv",
    "streamingmovies",

    "contract",
    "paperlessbilling",
    "paymentmethod",

    "tenure",
    "monthlycharges",
    "totalcharges"
];


/* =========================================================
   GET VALUE FROM INPUT
   ========================================================= */

function getValue(id) {
    return document.getElementById(id).value;
}


/* =========================================================
   COLLECT USER INPUT
   ========================================================= */

function collectCustomerData() {

    return {

        gender: getValue("gender"),

        partner: getValue("partner"),

        dependents: getValue("dependents"),


        phoneservice: getValue("phoneservice"),

        multiplelines: getValue("multiplelines"),


        internetservice: getValue("internetservice"),

        onlinesecurity: getValue("onlinesecurity"),

        onlinebackup: getValue("onlinebackup"),

        deviceprotection: getValue("deviceprotection"),

        techsupport: getValue("techsupport"),


        streamingtv: getValue("streamingtv"),

        streamingmovies: getValue("streamingmovies"),


        contract: getValue("contract"),

        paperlessbilling: getValue("paperlessbilling"),

        paymentmethod: getValue("paymentmethod"),


        tenure: Number(
            getValue("tenure")
        ),

        monthlycharges: Number(
            getValue("monthlycharges")
        ),

        totalcharges: Number(
            getValue("totalcharges")
        )
    };
}


/* =========================================================
   FRONTEND VALIDATION
   ========================================================= */

function validateForm() {

    let valid = true;


    /*
     * Check empty fields
     */

    fields.forEach((fieldName) => {

        const element =
            document.getElementById(fieldName);

        const wrapper =
            element.closest(".field");


        wrapper.classList.remove("invalid");


        if (element.value === "") {

            wrapper.classList.add("invalid");

            valid = false;
        }
    });


    /*
     * Numeric thresholds
     */

    const numericRules = [

        ["tenure", 0, 72],

        ["monthlycharges", 0, 150],

        ["totalcharges", 0, 10000]

    ];


    numericRules.forEach(
        ([id, min, max]) => {

            const element =
                document.getElementById(id);

            const wrapper =
                element.closest(".field");

            const value =
                Number(element.value);


            if (
                element.value === "" ||
                Number.isNaN(value) ||
                value < min ||
                value > max
            ) {

                wrapper.classList.add("invalid");

                valid = false;
            }
        }
    );


    if (!valid) {

        showError(
            "Please complete all fields and keep numeric values inside their allowed ranges."
        );
    }


    return valid;
}


/* =========================================================
   SHOW ERROR
   ========================================================= */

function showError(message) {

    errorMessage.textContent = message;

    errorCard.classList.remove("hidden");
}


/* =========================================================
   HIDE ERROR
   ========================================================= */

function hideError() {

    errorCard.classList.add("hidden");

    errorMessage.textContent = "";
}


/* =========================================================
   LOADING STATE
   ========================================================= */

function setLoading(isLoading) {

    predictButton.disabled = isLoading;

    predictButton.classList.toggle(
        "loading",
        isLoading
    );


    const buttonText =
        predictButton.querySelector(
            ".button-text"
        );


    if (isLoading) {

        buttonText.textContent =
            "Analyzing customer...";

    } else {

        buttonText.textContent =
            "Predict Customer Churn";
    }
}


/* =========================================================
   DISPLAY PREDICTION RESULT
   ========================================================= */

function displayResult(data) {

    const prediction =
        Number(data.prediction);


    const probability =
        Number(data.probability);


    /*
     * Validate probability
     */

    if (!Number.isFinite(probability)) {

        throw new Error(
            "The API returned an invalid probability."
        );
    }


    /*
     * Convert probability to percentage
     */

    const percentage =
        Math.max(
            0,
            Math.min(
                100,
                probability * 100
            )
        );


    /*
     * Show result card
     */

    resultCard.classList.remove(
        "hidden",
        "high-risk",
        "low-risk"
    );


    /*
     * Churn prediction
     */

    if (prediction === 1) {

        resultCard.classList.add(
            "high-risk"
        );


        resultTitle.textContent =
            "Customer is likely to churn";


        resultBadge.textContent =
            "HIGHER CHURN RISK";


    } else {

        resultCard.classList.add(
            "low-risk"
        );


        resultTitle.textContent =
            "Customer is unlikely to churn";


        resultBadge.textContent =
            "LOWER CHURN RISK";
    }


    /*
     * Probability
     */

    probabilityValue.textContent =
        `${percentage.toFixed(2)}%`;


    /*
     * Progress bar
     */

    progressBar.style.width =
        `${percentage}%`;


    /*
     * Prediction message
     */

    resultMessage.textContent =
        data.result ||
        (
            prediction === 1

                ? "The model predicts that this customer is likely to churn."

                : "The model predicts that this customer is unlikely to churn."
        );


    /*
     * Scroll to result
     */

    resultCard.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
}


/* =========================================================
   CHECK FASTAPI HEALTH
   ========================================================= */

async function checkApiHealth() {

    try {

        const response =
            await fetch(
                "/health",
                {
                    method: "GET",
                    cache: "no-store"
                }
            );


        if (!response.ok) {

            throw new Error(
                "API health check failed."
            );
        }


        apiStatus.classList.add(
            "online"
        );

        apiStatus.classList.remove(
            "offline"
        );


        statusText.textContent =
            "API Online";


    } catch (error) {

        apiStatus.classList.add(
            "offline"
        );

        apiStatus.classList.remove(
            "online"
        );


        statusText.textContent =
            "API Offline";
    }
}


/* =========================================================
   PREDICTION REQUEST
   ========================================================= */

form.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();


        hideError();


        /*
         * Validate input
         */

        if (!validateForm()) {

            return;
        }


        /*
         * Collect dynamic user input
         */

        const customerData =
            collectCustomerData();


        /*
         * Show loading state
         */

        setLoading(true);


        try {

            /*
             * Send user input to FastAPI
             */

            const response =
                await fetch(
                    "/predict",
                    {

                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                customerData
                            )
                    }
                );


            /*
             * Read JSON response
             */

            let data;


            try {

                data =
                    await response.json();

            } catch {

                throw new Error(
                    `Server returned an invalid response (HTTP ${response.status}).`
                );
            }


            /*
             * Handle FastAPI errors
             */

            if (!response.ok) {

                const detail =
                    data.detail ||
                    "The prediction API rejected the request.";


                throw new Error(detail);
            }


            /*
             * Display prediction
             */

            displayResult(data);


        } catch (error) {

            resultCard.classList.add(
                "hidden"
            );


            showError(
                error.message ||
                "Unable to connect to the prediction API."
            );


        } finally {

            /*
             * Remove loading state
             */

            setLoading(false);
        }
    }
);


/* =========================================================
   RESET FORM
   ========================================================= */

resetButton.addEventListener(
    "click",
    () => {

        /*
         * Restore HTML default values
         */

        form.reset();


        hideError();


        /*
         * Hide previous result
         */

        resultCard.classList.add(
            "hidden"
        );


        resultCard.classList.remove(
            "high-risk",
            "low-risk"
        );


        progressBar.style.width =
            "0%";


        /*
         * Remove validation errors
         */

        fields.forEach(
            (fieldName) => {

                const element =
                    document.getElementById(
                        fieldName
                    );


                element
                    .closest(".field")
                    .classList
                    .remove("invalid");
            }
        );
    }
);


/* =========================================================
   REMOVE VALIDATION ERROR WHEN USER CHANGES INPUT
   ========================================================= */

fields.forEach(
    (fieldName) => {

        const element =
            document.getElementById(
                fieldName
            );


        element.addEventListener(
            "input",
            () => {

                element
                    .closest(".field")
                    .classList
                    .remove("invalid");


                hideError();
            }
        );


        element.addEventListener(
            "change",
            () => {

                element
                    .closest(".field")
                    .classList
                    .remove("invalid");


                hideError();
            }
        );
    }
);


/* =========================================================
   INITIAL API HEALTH CHECK
   ========================================================= */

checkApiHealth();


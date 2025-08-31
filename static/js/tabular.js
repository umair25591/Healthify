AOS.init();

$(document).ready(function () {

    let currentStep = 0;
    let activeForm = $('.survey-form.active-form')[0];
    let questions = $(activeForm).find('.question');
    const nextBtn = $('#nextBtn');
    const reportBox = $('#reportBox');
    const progressBar = $('#progress-bar');
    const surveyContent = $('#survey-content');

    const assistantPrompts = {
        'default': "Welcome! Please select a health category above to begin your assessment.",
        
        // Kidney Disease
        'blood_pressure': "Blood pressure is the force of blood pushing against artery walls. It's a key indicator of kidney health.",
        'specific_gravity': "Specific gravity measures urine concentration. It shows how well the kidneys are diluting your urine.",
        'albumin': "Albumin is a protein. Its presence in urine can be an early sign of kidney damage.",
        'blood_glucose_random': "This measures your blood sugar at a random time. High levels can affect your kidneys.",
        
        // Heart Disease
        'age': "Age is a primary risk factor for many heart conditions. Let's start here.",
        'chest_pain_type': "The type of chest discomfort you experience can give important clues about its potential cause.",
        'cholesterol': "Cholesterol is a fatty substance in your blood. High levels of certain types can lead to blockages.",
        'max_heart_rate': "This is the highest heart rate you achieved during a stress test, indicating your heart's response to exercise.",
        'exercise_angina': "Angina is chest pain caused by reduced blood flow. Experiencing it during exercise is a significant symptom.",
        
        // Diabetes
        'pregnancies': "The number of past pregnancies can sometimes be a factor in metabolic health and diabetes risk.",
        'glucose': "This measures the amount of sugar in your bloodstream. It's a direct indicator of how your body processes sugar.",
        'insulin': "Insulin is the hormone that regulates blood sugar. This test measures how much insulin your body produces after a meal.",
        'bmi': "Your Body Mass Index (BMI) helps gauge if your weight is healthy in proportion to your height.",
        'diabetes_pedigree': "This function is a score that represents the likelihood of diabetes based on your family history."
    };

    const infoContent = {
        'default': "<p>Your health is important. This tool provides an AI-based analysis, not a medical diagnosis. Always consult a healthcare professional.</p>",
       
        // Kidney Disease
        'blood_pressure': "<h5>Normal BP</h5><p>A normal blood pressure reading is typically around 120/80 mm Hg. The top number (systolic) measures pressure during heartbeats, and the bottom (diastolic) measures pressure between beats.</p>",
        'specific_gravity': "<h5>Concentration</h5><p>A healthy range for urine specific gravity is generally between 1.010 and 1.025. Deviations can indicate hydration issues or kidney problems.</p>",
        'albumin': "<h5>Protein in Urine</h5><p>Normally, you should have little to no albumin in your urine. A level of '0' is ideal. Higher numbers indicate increasing levels of protein leakage.</p>",
        'blood_glucose_random': "<h5>Random Glucose Test</h5><p>A result of 200 mg/dL or higher can be an indicator of diabetes, a condition closely linked with chronic kidney disease.</p>",
       
        // Heart Disease
        'age': "<h5>Age as a Factor</h5><p>The risk of heart disease increases for everyone as they get older. For men, the risk increases after 45. For women, it increases after 55.</p>",
        'chest_pain_type': "<h5>Angina Pectoris</h5><p>Typical angina is often described as a pressure or squeezing in the chest, which can be a primary symptom of coronary artery disease.</p>",
        'cholesterol': "<h5>Good vs. Bad</h5><p>Total cholesterol below 200 mg/dL is desirable. It's composed of LDL ('bad') and HDL ('good') cholesterol. High LDL is a major risk factor.</p>",
        'max_heart_rate': "<h5>Target Zones</h5><p>Your theoretical maximum heart rate is often estimated as 220 minus your age. Doctors use your actual achieved rate to evaluate cardiac health.</p>",
        'exercise_angina': "<h5>Stable vs. Unstable</h5><p>Angina that occurs predictably during exertion is called 'stable angina'. It's a warning sign that the heart isn't getting enough oxygen during high-demand periods.</p>",
        
        // Diabetes
        'pregnancies': "<h5>Gestational Diabetes</h5><p>Some women develop high blood sugar during pregnancy. While it usually resolves after birth, it can increase the risk of developing Type 2 diabetes later in life.</p>",
        'glucose': "<h5>Fasting vs. Post-Meal</h5><p>A fasting blood glucose level below 100 mg/dL is normal. A level of 126 mg/dL or higher on two separate tests indicates diabetes.</p>",
        'insulin': "<h5>Insulin Resistance</h5><p>High insulin levels can indicate 'insulin resistance,' where the body's cells don't respond effectively to insulin. This is a precursor to Type 2 diabetes.</p>",
        'bmi': "<h5>Weight and Risk</h5><p>A BMI over 25 is considered overweight, and over 30 is obese. Excess weight, particularly around the abdomen, is a major risk factor for Type 2 diabetes.</p>",
        'diabetes_pedigree': "<h5>Genetics Matter</h5><p>Family history plays a strong role in diabetes risk. This 'pedigree' value quantifies that genetic risk based on your relatives' history with the disease.</p>"
    };


    function updateFormState() {
        activeForm = $('.survey-form.active-form')[0];
        questions = $(activeForm).find('.question');
        currentStep = 0;
        showStep(currentStep);
        updateProgressBar();
    }

    function updateProgressBar() {
        const percent = ((currentStep + 1) / questions.length) * 100;
        progressBar.css('width', `${percent}%`);
    }

    function showStep(step) {
        let currentQuestionName = 'default';
        questions.each(function (index, q) {
            if (index === step) {
                $(q).removeClass('exiting').addClass('active');
                currentQuestionName = $(q).find('input, select').attr('name');
            } else {
                $(q).removeClass('active');
            }
        });

        $('#ai-prompt-text').text(assistantPrompts[currentQuestionName] || assistantPrompts['default']);
        $('#info-content-box').html(infoContent[currentQuestionName] || infoContent['default']);

        nextBtn.text(step === questions.length - 1 ? "Analyze Results" : "Next Question");
        updateProgressBar(); // Update progress bar on each step
    }

    $('.tab-btn').on('click', function () {
        const formId = $(this).data('form');
        $('.tab-btn').removeClass('active-tab');
        $(this).addClass('active-tab');
        $('.survey-form').removeClass('active-form');
        $(`#${formId}`).addClass('active-form');
        reportBox.hide();
        surveyContent.show();
        updateFormState();
    });

    nextBtn.on('click', function () {
        const currentQuestion = $(questions[currentStep]);
        const input = currentQuestion.find('input, select')[0];
        if (!input.checkValidity()) {
            input.reportValidity();
            return;
        }
        if (currentStep < questions.length - 1) {
            currentQuestion.addClass('exiting');
            currentStep++;
            setTimeout(() => {
                showStep(currentStep);
            }, 400);
        } else {
            generateReport();
        }
    });

    function generateReport() {
        
    }

    updateFormState();
});
$(document).ready(function () {

    // --- SWITCHER TAB TOGGLING ---
    $(".switcher__input").on("change", function () {
        // Remove active from all tab panes
        $(".tab-pane").removeClass("active");

        // Get target ID from data-target attribute
        const target = $(this).data("target");
        if (target) {
            $("#" + target).addClass("active");
        }

        // Update active styling on radios (optional, for CSS)
        $(".switcher__input").removeClass("active");
        $(this).addClass("active");
    });

    // --- FORM COMPLETION CHECK ---
    function checkForms() {
        const $forms = $('form');
        let allModulesInActiveFormComplete = true;

        $forms.each(function () {
            const $form = $(this);
            const $modules = $form.find('.row > div[id]');

            if ($modules.length === 0) {
                if ($form.closest('.tab-pane').hasClass('active')) {
                    allModulesInActiveFormComplete = false;
                }
                return;
            }

            $modules.each(function () {
                const $module = $(this);
                const $card = $module.find('.form-cards');
                if ($card.length === 0) return;

                const $requiredInputs = $module.find('[required]');
                let isModuleComplete = true;

                $requiredInputs.each(function () {
                    if (!$(this).val()) {
                        isModuleComplete = false;
                        return false;
                    }
                });

                $card.toggleClass('complete', isModuleComplete);

                if ($form.closest('.tab-pane').hasClass('active') && !isModuleComplete) {
                    allModulesInActiveFormComplete = false;
                }
            });
        });

        // You can use `allModulesInActiveFormComplete` to enable/disable submit, etc.
    }

    $('form').on('input', checkForms);
    checkForms();

    // --- LIQUID SWITCHER DIRECTION TRACKING ---
    const switcher = document.getElementById("diseaseSwitcher");
    const radios = switcher.querySelectorAll('input[type="radio"]');
    let previous = switcher.querySelector('input:checked')?.getAttribute("c-option");

    radios.forEach(radio => {
      radio.addEventListener("change", () => {
        const current = radio.getAttribute("c-option");
        switcher.setAttribute("c-previous", previous);
        previous = current;
      });
    });
});

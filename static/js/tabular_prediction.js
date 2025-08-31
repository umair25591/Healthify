$(document).ready(function () {
    $(".disease-toggle").on("click", function () {

        $(".disease-toggle").removeClass("active");
        $(this).addClass("active");

        $(".tab-pane").removeClass("active");

        let target = $(this).data("target");
        $("#" + target).addClass("active");
    });
});

$(function () {

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

    }

    $('form').on('input', checkForms);
    checkForms();
});
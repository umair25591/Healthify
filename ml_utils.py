import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def generate_shap_plot_base64(explainer, prediction_df, class_index=1):
    try:
        row = prediction_df.iloc[[0]]
        try:
            shap_values = explainer.shap_values(row)
            expected_value = explainer.expected_value
        except Exception:
            shap_values = explainer(row)
            expected_value = shap_values.base_values

        if isinstance(shap_values, shap._explanation.Explanation):
            if shap_values.values.ndim == 3:
                shap_values_for_plot = shap_values.values[0, class_index, :].ravel()
                expected_value = shap_values.base_values[0, class_index]
                feature_names = shap_values.feature_names
            elif shap_values.values.ndim == 2:
                shap_values_for_plot = shap_values.values[0, :].ravel()
                expected_value = shap_values.base_values[0]
                feature_names = shap_values.feature_names
            else:
                shap_values_for_plot = shap_values.values.ravel()
                expected_value = shap_values.base_values
                feature_names = shap_values.feature_names
        elif isinstance(shap_values, list):
            shap_values_for_plot = np.array(shap_values[class_index][0]).ravel()
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[class_index]
            feature_names = prediction_df.columns.tolist()
        else:
            shap_values_for_plot = np.array(shap_values[0]).ravel()
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[0]
            feature_names = prediction_df.columns.tolist()

        n_features = min(len(feature_names), len(shap_values_for_plot))
        shap_values_for_plot = shap_values_for_plot[:n_features]
        feature_names = feature_names[:n_features]
        feature_values = row.iloc[0, :n_features].values

        fig = shap.force_plot(
            expected_value,
            shap_values_for_plot,
            feature_values,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            text_rotation=15
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)

        return image_base64, shap_values_for_plot

    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
        return None, None


def generate_insights(shap_values, feature_names, input_data, top_n=3, class_index=1):
    try:
        if shap_values is None:
            return [{"class": "high-risk", "text": "Could not generate insights due to an analysis error."}]

        if isinstance(shap_values, shap._explanation.Explanation):
            if shap_values.values.ndim == 3:
                shap_row = shap_values.values[0, class_index, :].ravel()
                feature_names = shap_values.feature_names
            elif shap_values.values.ndim == 2: 
                shap_row = shap_values.values[0, :].ravel()
                feature_names = shap_values.feature_names
            else:
                shap_row = shap_values.values.ravel()
                feature_names = shap_values.feature_names
        elif isinstance(shap_values, list):
            shap_row = np.array(shap_values[class_index][0]).ravel()
        else:
            shap_row = np.array(shap_values[0]).ravel()

        n_features = min(len(feature_names), len(shap_row))
        shap_row = shap_row[:n_features]
        feature_names = feature_names[:n_features]

        feature_impacts = sorted(
            zip(feature_names, shap_row),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        insights = []
        for feature, shap_val in feature_impacts[:top_n]:
            user_value = input_data.get(feature, "N/A")
            insight = {}

            if shap_val > 0.05:
                insight["class"] = "high-risk"
                insight["text"] = (
                    f"Your <strong>{feature}</strong> value (<em>{user_value}</em>) "
                    f"was a primary factor <strong>increasing</strong> your risk score."
                )
            elif shap_val < -0.05:
                insight["class"] = "low-risk"
                insight["text"] = (
                    f"Your <strong>{feature}</strong> value (<em>{user_value}</em>) "
                    f"played a key role in <strong>lowering</strong> your risk score."
                )
            else:
                insight["class"] = "secondary"
                insight["text"] = (
                    f"Your <strong>{feature}</strong> value (<em>{user_value}</em>) "
                    f"had a minor impact on the result."
                )

            insights.append(insight)

        return insights

    except Exception as e:
        print(f"Error generating insights: {e}")
        return [{"class": "high-risk", "text": "Could not generate insights due to an analysis error."}]

import shap
import matplotlib.pyplot as plt

# model: Model of distilated policy
# X_test: Test dataset
# SHAP_PLOT_TYPE: Decision_PLOT_TYPE
# actions: actions of test_dataset


def cal_shap(model, X, SHAP_PLOT_TYPE, actions):
    path = save_path()
    explainer = shap.TreeExplainer(model, X)

    # Cal Shap Values
    shap_values = explainer(X)

    # Makes Shap plots
    for x in range(len(actions)):
        # shap_values[:, :, x] means xth class's shap value
        shap.summary_plot(shap_values[:, :, x], X,
                          plot_type=SHAP_PLOT_TYPE, show=False)
        p = f"{path}/Action's{actions[x]}_shap.png"
        plt.savefig(p)
        print(p)
        plt.close()


def save_path():
    import os
    import shutil

    # Current working path
    current_path = os.path.dirname(__file__)

    # Parent folder path
    parent_path = os.path.dirname(current_path)

    # New folder path
    summary_plot_path = os.path.join(parent_path, "summary_plot")

    # Remove new folder if it already exists
    if os.path.exists(summary_plot_path):
        shutil.rmtree(summary_plot_path)

    # Create a new folder
    os.mkdir(summary_plot_path)

    return summary_plot_path

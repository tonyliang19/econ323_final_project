# helpers for plots

# consider install seaborn if not exists yet
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# eda plot using seaborn, default uses hist for histogram
def eda_plot(data, title, option, **kwargs):
    # common parameters to control title position and size
    fontsize=12
    ypos=-0.03
    
    # option to disply heatmap
    if option == "heatmap":
        # Set the size of the figure
        plt.figure(figsize=(15, 8))

        # Compute the correlation matrix for the Boston dataset 
        # and create a heatmap of the absolute correlation coefficients
        sns.heatmap(data.corr().abs(), annot=True)

        # Add a title to the plot
        # special position for heatmap
        plt.title(title, y = ypos - 0.07,fontsize=fontsize)
        plt.show()
    # if not heatmap, then its either of histogram or boxplot
    else:
        # Set up a figure with subplots
        fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(14, 7))
        axs = axs.flatten()

        # Iterate over the columns in the Boston dataset and plot their distribution using seaborn
        if option == "hist":
            for i, col in enumerate(data.columns):
                sns.histplot(data=data[col], ax=axs[i], kde=True)
            fig.suptitle(title, y = ypos, fontsize=fontsize)
        if option == "box":
            for i, col in enumerate(data.columns):
                sns.boxplot(y=data[col], ax=axs[i])
            fig.suptitle(title, y = ypos, fontsize=fontsize)

        # Adjust the spacing of the subplots and display the plot and add title
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
        plt.show()

        
# helper to plot the -log alphas vs metrics for cv in LASSO
def plot_alpha_metrics(title, df, colors=None):
    # metric names
    metrics = ["root_mean_squared_error", "mean_squared_error", "mean_absolute_error"]
    # color choices
    if colors is None:
        colors = ['#165aa7', '#f47915',  '#007030'] # replacement colors []'#bb60d5', '#f47915', '#06ab54', '#002070', '#b27d12', '#007030']
    # convert to log10 of alphas for better interpretability
    df["log_alpha"] = -np.log10(df["alpha"])
    
    # plot out the metrics
    fig, ax = plt.subplots(figsize=(10,6))
    
    # initial position for x and y
    i = 25
    for k, m in enumerate(metrics):
        df.plot(x="log_alpha", y=m, ax=ax, c=colors[k])
        if m.__contains__("mean_squared"):
            x_pos = df.log_alpha[85]
            y_pos = df[m][25]
        else:
            x_pos = df.log_alpha[30 + i]
            y_pos = df.log_alpha[50 + i]
        # replace _ with underscore and capitalize beginning of each word
        ax.annotate(m.replace("_", " ").title(), (x_pos, y_pos), color=colors[k])
        # increment to have different position
        i = i + 25
    ax.set_xlabel(r"$-\log(\alpha)$")
    ax.set_ylabel("Metric value")
    ax.get_legend().remove()
    plt.suptitle(title, y = -0.001)

# helpers for plots

# consider install seaborn if not exists yet
import seaborn as sns
import matplotlib.pyplot as plt

# eda plot using seaborn, default uses hist for histogram
def eda_plot(data, title, option):
    # option to disply heatmap
    if option == "heatmap":
        # Set the size of the figure
        plt.figure(figsize=(20, 12))

        # Compute the correlation matrix for the Boston dataset 
        # and create a heatmap of the absolute correlation coefficients
        sns.heatmap(data.corr().abs(), annot=True)

        # Add a title to the plot
        plt.title(title)
        plt.show()
    # if not heatmap, then its either of histogram or boxplot
    else:
        # Set up a figure with subplots
        fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
        axs = axs.flatten()

        # Iterate over the columns in the Boston dataset and plot their distribution using seaborn
        if option == "hist":
            for i, col in enumerate(data.columns):
                sns.histplot(data=data[col], ax=axs[i], kde=True)
            fig.suptitle(title)
        if option == "box":
            for i, col in enumerate(data.columns):
                sns.boxplot(y=data[col], ax=axs[i])
            fig.suptitle(title)

        # Adjust the spacing of the subplots and display the plot and add title
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
        plt.show()
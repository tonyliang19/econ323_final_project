# helpers for plots

# consider install seaborn if not exists yet
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
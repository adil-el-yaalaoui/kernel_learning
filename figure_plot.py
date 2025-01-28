import matplotlib.pyplot as plt


def plot_solutions(fig,noise_levels:list,training_sizes:list,classification_errors:dict,rkhs_norms:dict):

    for i, noise in enumerate(noise_levels):
        ax = fig.add_subplot(3, 2, 2 * i + 1)  # 3 rows, left column (1, 3, 5)

        ax.plot(training_sizes, classification_errors[noise]["interpolated"], label="Interpolated", linestyle="--", marker="s", color="blue")
        ax.plot(training_sizes, classification_errors[noise]["overfitted"], label="Overfitted", linestyle="-", marker="o", color="red")
        ax.plot(training_sizes, classification_errors[noise]["bayes"], label="Bayes ", linestyle=":", marker="x", color="green")

        ax.set_xlabel("Training Size")
        ax.set_ylabel("Classification Error (%)")
        ax.set_title(f"Classification Error (Noise={int(noise * 100)}%)")
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    ax_rkhs = fig.add_subplot(1, 2, 2)

    colors=["blue", "orange", "purple"]
    for i,noise in enumerate(noise_levels):
        ax_rkhs.plot(training_sizes, rkhs_norms[noise]["interpolated"], label=f"Interpolated (Noise={int(noise * 100)}%)", color=colors[i], linestyle="--", marker="s")
        ax_rkhs.plot(training_sizes, rkhs_norms[noise]["overfitted"], label=f"Overfitted (Noise={int(noise * 100)}%)", color=colors[i], linestyle="-", marker="o")

    ax_rkhs.set_xlabel("Training Size")
    ax_rkhs.set_ylabel("RKHS Norm")
    ax_rkhs.set_title("Evolution of RKHS Norms")
    ax_rkhs.legend()
    ax_rkhs.spines['top'].set_visible(False)
    ax_rkhs.spines['right'].set_visible(False)
    ax_rkhs.grid(False)

    plt.tight_layout()
    plt.show()


def plot_solutions_nn(fig,noise_levels:list,training_sizes:list,classification_errors:dict,rkhs_norms:dict):

    for i, noise in enumerate(noise_levels):
        ax = fig.add_subplot(3, 2, 2 * i + 1)  # 3 rows, left column (1, 3, 5)

        ax.plot(training_sizes, classification_errors[noise]["NN"], label="NN", linestyle="--", marker="s", color="blue")

        ax.set_xlabel("Training Size")
        ax.set_ylabel("Classification Error (%)")
        ax.set_title(f"Classification Error (Noise={int(noise * 100)}%)")
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    ax_rkhs = fig.add_subplot(1, 2, 2)

    colors=["blue", "orange", "purple"]
    for i,noise in enumerate(noise_levels):
        ax_rkhs.plot(training_sizes, rkhs_norms[noise]["NN"], label=f"NN (Noise={int(noise * 100)}%)", color=colors[i], linestyle="--", marker="s")

    ax_rkhs.set_xlabel("Training Size")
    ax_rkhs.set_ylabel("RKHS Norm")
    ax_rkhs.set_title("Evolution of RKHS Norms")
    ax_rkhs.legend()
    ax_rkhs.spines['top'].set_visible(False)
    ax_rkhs.spines['right'].set_visible(False)
    ax_rkhs.grid(False)

    plt.tight_layout()
    plt.show()

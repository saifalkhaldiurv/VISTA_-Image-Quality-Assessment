import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

def plot_images_from_subfolders(folder_path, output_file, csv_file, colors, cols_arrange=None, ntop=10, resize=10):
    subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    fig, axs = plt.subplots(nrows=len(subfolders), ncols=(ntop+1), figsize=((ntop+1)*resize, len(subfolders)*resize))
    df = pd.read_csv(os.path.join(folder_path, csv_file))
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
    df = df[cols_arrange] if cols_arrange else df
    for i, subfolder in enumerate(subfolders):
        images = sorted([os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.png') or f.endswith('.jpg')], key=lambda x: int(x.split('_')[-3].split('.')[0]))[:ntop]
        axs[i, 0].axis('off')
        axs[i, 0].text(0.5, 0.5, os.path.basename(subfolder), horizontalalignment='center', verticalalignment='center', fontsize=12*resize//2, color=colors[i])
        for j, image in enumerate(images):
            img = Image.open(image)
            axs[i, j+1].imshow(img)
            axs[i, j+1].axis('off')

            image_name = os.path.basename(image)[9:]
            # print(image_name)
            values = df[df['Image Name'] == image_name][df.columns[1:]].values[0]

            ax_inset = axs[i, j+1].inset_axes([0.75, 0.75, 0.25, 0.25], facecolor='none')
            ax_inset.bar(range(len(values)), values, color=colors)
            # ax_inset.set_xticks(range(len(values)))
            # ax_inset.set_xticklabels(df.columns[1:], rotation=90)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            ax_inset.set_frame_on(False)

            # import matplotlib.patheffects as path_effects
            # for label in ax_inset.get_xticklabels():
            #     label.set_fontsize(12*2)
            #     label.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)

colors = ['forestgreen', 'dodgerblue', 'orangered']
folder_path = 'plot/good_illumination-bright_illumination-dark/vgg16patch16/8_rot'
output_file = 'titan/plot_concept_top10.png'
csv_file = 'vals_all.csv'
cols_arrange = ['Image Name', 'good', 'illumination-bright', 'illumination-dark']
plot_images_from_subfolders(folder_path, output_file, csv_file, colors, cols_arrange, ntop=5)

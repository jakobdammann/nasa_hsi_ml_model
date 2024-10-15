import matplotlib.pyplot as plt
import numpy as np
import tifffile
import os
from matplotlib import widgets
import matplotlib.gridspec as gridspec
from torchmetrics.image import RelativeAverageSpectralError
from torchmetrics.functional.image import relative_average_spectral_error
from torchmetrics.image import SpectralAngleMapper
import torch
import config as c

# Folder paths for images
thorlabs_image_folder = 'test/imgs'
cubert_image_folder = 'test/imgs'
gen_image_folder = 'test/imgs'


# List available files in each folder
thorlabs_files = sorted([f for f in os.listdir(thorlabs_image_folder) if f.endswith(".tif") and f.startswith("tl_raw")])
cubert_files = sorted([f for f in os.listdir(cubert_image_folder) if f.endswith(".tif") and f.startswith("cb_raw")])
gen_files = sorted([f for f in os.listdir(gen_image_folder) if f.endswith(".tif") and f.startswith("tl_gen")])


# Initial selections
current_img_index = 0
current_tl_file = thorlabs_files[current_img_index%len(thorlabs_files)]
current_cb_file = cubert_files[current_img_index%len(cubert_files)]
current_gen_file = gen_files[current_img_index%len(gen_files)]

current_channel = 53
wavelengths = np.linspace(450, 850, 106)  # Assuming 106 channels from 450-850 nm
spectra_cb = []
spectra_gen = []

current_pol = 0

# List to keep track of multiple selected regions
selected_regions = []

# Load CB image from file path
def load_cb_image(cb_file):
    cb_image_path = os.path.join(cubert_image_folder, cb_file)
    # Load Cubert image
    cb_image = tifffile.imread(cb_image_path)
    return cb_image

# Load TL image from file path
def load_tl_image(tl_file):
    tl_image_path = os.path.join(thorlabs_image_folder, tl_file)
    # Load Cubert image
    tl_image = tifffile.imread(tl_image_path)
    return tl_image

# Load Gen image from file path
def load_gen_image(gen_file):
    gen_image_path = os.path.join(gen_image_folder, gen_file)
    # Load Cubert image
    gen_image = tifffile.imread(gen_image_path)
    return gen_image

# Update the CB plot with the selected image and channel
def update_cb_plot(cb_file, channel):
    global ax_cb, fig  # Make sure cb_image is accessible in the callback
    global colorbar_cb
    global cb_image
    # Clear the current plot
    ax_cb.clear()
    # Plot selected channel of Cubert image
    img = cb_image[channel, :, :]
    im_cb = ax_cb.imshow(img, cmap='viridis')
    wavelength = wavelengths[channel]
    ax_cb.set_title(f"Ground Truth: {cb_file} (Channel {channel}/{cb_image.shape[0]-1}, {wavelength:.1f} nm)")
    ax_cb.annotate(f"Channel Stats: \nSNR: {snr(img):.2f}, Min: {np.min(img):.2f}, Max: {np.max(img):.2f}, Avg: {np.mean(img):.2f}, Std: {np.std(img):.2f}", 
                   (-0.05,-0.18), xycoords='axes fraction')
    # create or update colorbars
    if colorbar_cb != None:
        colorbar_cb.update_normal(im_cb)
    else:
        colorbar_cb = plt.colorbar(im_cb)
    # Redraw the figure
    fig.canvas.draw_idle()

# Update the TL plot with the selected images and channel
def update_tl_plot(tl_file, pol):
    global ax_tl, ax_cb, fig  # Make sure cb_image is accessible in the callback
    global colorbar_tl
    global tl_image

    # Clear the current plot
    ax_tl.clear()

    # Plot Thorlabs image
    pol_index = int(pol / 45)
    img = tl_image[pol_index]
    im_tl = ax_tl.imshow(img, cmap='viridis')
    ax_tl.set_title(f"Input: {tl_file} (P={pol})")
    ax_tl.annotate(f"Channel Stats: \nSNR: {snr(img):.2f}, Min: {np.min(img):.2f}, Max: {np.max(img):.2f}, Avg: {np.mean(img):.2f}, Std: {np.std(img):.2f}", 
                   (-0.05,-0.18), xycoords='axes fraction')

    # create or update colorbars
    if colorbar_tl != None:
        colorbar_tl.update_normal(im_tl)
    else:
        colorbar_tl = plt.colorbar(im_tl)
    
    # Redraw the figure
    fig.canvas.draw_idle()

# Update ...
def update_gen_plot(gen_file, channel):
    global ax_gen, fig  # Make sure cb_image is accessible in the callback
    global colorbar_gen
    global gen_image
    # Clear the current plot
    ax_gen.clear()
    # Plot selected channel of Cubert image
    img = gen_image[channel, :, :]
    im_gen = ax_gen.imshow(img, cmap='viridis')
    wavelength = wavelengths[channel]
    ax_gen.set_title(f"Generated: {gen_file} (Channel {channel}/{gen_image.shape[0]-1}, {wavelength:.1f} nm)")
    ax_gen.annotate(f"Channel Stats: \nSNR: {snr(img):.2f}, Min: {np.min(img):.2f}, Max: {np.max(img):.2f}, Avg: {np.mean(img):.2f}, Std: {np.std(img):.2f}", 
                   (-0.05,-0.18), xycoords='axes fraction')
    # create or update colorbars
    if colorbar_gen != None:
        colorbar_gen.update_normal(im_gen)
    else:
        colorbar_gen = plt.colorbar(im_gen)
    # Redraw the figure
    fig.canvas.draw_idle()

# Change the Thorlabs file
def change_thorlabs_file(text):
    global current_tl_file, tl_image
    if text in thorlabs_files:
        current_tl_file = text
        tl_image = load_tl_image(current_tl_file)
        update_tl_plot(current_tl_file, current_pol)
    else:
        print(f"File '{text}' not found in Thorlabs folder.")

# Change the Cubert file
def change_cubert_file(text):
    global current_cb_file, channel_slider, cb_image
    if text in cubert_files:
        current_cb_file = text
        cb_image = load_cb_image(current_cb_file)
        update_cb_plot(current_cb_file, current_channel)
        channel_slider.valmax = load_cb_image(current_cb_file).shape[2] - 1
        channel_slider.set_val(current_channel)  # Update channel slider to new file's channel count
    else:
        print(f"File '{text}' not found in Cubert folder.")

# Change the Gen file
def change_gen_file(text):
    global current_gen_file, channel_slider, gen_image
    if text in gen_files:
        current_gen_file = text
        gen_image = load_gen_image(current_gen_file)
        update_gen_plot(current_gen_file, current_channel)
    else:
        print(f"File '{text}' not found in Gen folder.")

# Change ID with text field
def change_img_id(id):
    change_thorlabs_file("tl_raw_" + str(id) + ".tif")
    change_cubert_file("cb_raw_" + str(id) + ".tif")
    change_gen_file("tl_gen_" + str(id) + ".tif")

# Change the channel
def change_channel(val):
    global current_channel
    current_channel = int(val)
    update_cb_plot(current_cb_file, current_channel)
    update_gen_plot(current_gen_file, current_channel)

# Get color from a wavelength
def get_color_from_wavelength(wavelength):
    if 380 <= wavelength < 450:
        return "Violet"
    elif 450 <= wavelength < 500:
        return "Blue"
    elif 500 <= wavelength < 570:
        return "Green"
    elif 570 <= wavelength < 590:
        return "Yellow"
    elif 590 <= wavelength < 620:
        return "Orange"
    elif 620 <= wavelength <= 750:
        return "Red"
    else:
        return "Out of Visible Range"

# Change the shown TL polarisation
def change_pol(val):
    global current_pol
    current_pol = int(val)
    update_tl_plot(current_tl_file, current_pol)

# Show next image
def next_image(_):
    global cb_image, tl_image, gen_image, current_img_index
    current_img_index += 1
    current_tl_file = thorlabs_files[current_img_index%len(thorlabs_files)]
    current_cb_file = cubert_files[current_img_index%len(cubert_files)]
    current_gen_file = gen_files[current_img_index%len(gen_files)]
    tl_image = load_tl_image(current_tl_file)
    cb_image = load_cb_image(current_cb_file)
    gen_image = load_gen_image(current_gen_file)
    update_tl_plot(current_tl_file, current_pol)
    update_cb_plot(current_cb_file, current_channel)
    update_gen_plot(current_gen_file, current_channel)
    print(f"Showing next images: {current_tl_file} (TL), {current_cb_file} (CB), {current_gen_file} (Gen)")

# Show prev image
def prev_image(_):
    global cb_image, tl_image, gen_image, current_img_index
    current_img_index -= 1
    current_tl_file = thorlabs_files[current_img_index%len(thorlabs_files)]
    current_cb_file = cubert_files[current_img_index%len(cubert_files)]
    current_gen_file = gen_files[current_img_index%len(gen_files)]
    tl_image = load_tl_image(current_tl_file)
    cb_image = load_cb_image(current_cb_file)
    gen_image = load_gen_image(current_gen_file)
    update_tl_plot(current_tl_file, current_pol)
    update_cb_plot(current_cb_file, current_channel)
    update_gen_plot(current_gen_file, current_channel)
    print(f"Showing previous images: {current_tl_file} (TL), {current_cb_file} (CB), {current_gen_file} (Gen)")

def clear_spectra(_):
    ax_spec_cb.clear()
    
# Calc SNR
def snr(img, axis=None, ddof=0):
    img = np.asanyarray(img)
    m = img.mean(axis)
    sd = img.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def avrg_metrics():
    total_rase = 0
    total_SRE = 0
    num_images = len(gen_files)
    
    for i in range(num_images):
        # Load the current Cubert and generated images
        current_cb_file = cubert_files[i]
        current_gen_file = gen_files[i]
        
        cb_image = load_cb_image(current_cb_file)
        gen_image = load_gen_image(current_gen_file)
        
        # Calculate RASE & SRE for the current image  
        rase = RASE(cb_image, gen_image)
        sre = SRE(gen_image, cb_image)
        total_rase += rase.item()
        total_SRE += sre.item()
        
    # Compute the average RASE over all images
    average_rase = total_rase / num_images
    average_SRE = total_SRE / num_images
    return average_rase, average_SRE

def RASE(cb_img, gen_img):
    rmse = np.sqrt(np.mean((cb_img - gen_img)**2, axis=(1,2)))
    rase = np.sqrt(np.mean(rmse**2)) * 100 / np.mean(cb_img)
    return rase

def SRE(I_pred, I_true):
    # Calculate the SRE for each pixel (x, y) and wavelength (λ)
    # Avoid division by zero by adding a small epsilon value to the denominator
    epsilon = 1e-8
    sre = ((I_pred - I_true) / (I_true + epsilon)) * 100 # Element-wise SRE in percentage
    # Average over all (x, y, λ)
    sre_avg = np.mean(sre, axis = (0,1,2))    
    return sre_avg


# Callback function for region selection
def onselect(eclick, erelease):
    global cb_image, gen_image, spectra_cb, spectra_gen

    # Get the coordinates of the rectangle
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    # Ensure coordinates are in the correct order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # Extract the reflectance values for the selected area
    selected_area_cb = cb_image[:, y1:y2+1, x1:x2+1]
    selected_area_gen = gen_image[:, y1:y2+1, x1:x2+1]

    # Calculate the average reflectance values for the selected area
    area_val_cb = np.mean(selected_area_cb, axis=(1, 2))
    area_val_gen = np.mean(selected_area_gen, axis=(1, 2))

    # Store the normalized reflectance values
    spectra_cb = area_val_cb
    spectra_gen = area_val_gen

    # Plot the spectra comparison
    plot_spectra_comparison(spectra_cb, spectra_gen)

# Plot both ground truth (HS) and prediction (Gen) spectra for comparison
def plot_spectra_comparison(spectrum_cb, spectrum_gen):
    global fig, ax_spec_cb

    # Clear the current plot
    ax_spec_cb.clear()

    # Plot the ground truth (HS) spectrum
    ax_spec_cb.plot(wavelengths, spectrum_cb, label='Ground Truth', color='blue')

    # Plot the generated image spectrum
    ax_spec_cb.plot(wavelengths, spectrum_gen, label='Generated Image', color='red')

    # Find the peak wavelength for the ground truth
    peak_index_cb = np.argmax(spectrum_cb)
    peak_wavelength_cb = wavelengths[peak_index_cb]
    color_cb = get_color_from_wavelength(peak_wavelength_cb)

    # Find the peak wavelength for the generated image
    peak_index_gen = np.argmax(spectrum_gen)
    peak_wavelength_gen = wavelengths[peak_index_gen]
    color_gen = get_color_from_wavelength(peak_wavelength_gen)

    # Annotate peak wavelengths
    ax_spec_cb.text(0.95, 0.85, f'Peak (HS): {peak_wavelength_cb:.1f} nm\nColor: {color_cb}', transform=ax_spec_cb.transAxes, fontsize=10, color='blue', verticalalignment='top', horizontalalignment='right')
    ax_spec_cb.text(0.95, 0.75, f'Peak (Gen): {peak_wavelength_gen:.1f} nm\nColor: {color_gen}', transform=ax_spec_cb.transAxes, fontsize=10, color='red', verticalalignment='top', horizontalalignment='right')

    # Add labels and legend
    ax_spec_cb.set_xlabel("Wavelength (nm)")
    ax_spec_cb.set_ylabel("Intensity")
    ax_spec_cb.set_title(f"Spectrum Comparison for Selected Area (Average SRE: {SRE(gen_image, cb_image):.3f})")
    ax_spec_cb.legend()

    # Redraw the figure
    fig.canvas.draw_idle()

# Plot all spectracs (works vor CB and Gen)
def plot_spectra(ax_spec_cb, spectra_cb, selected_area_cb):
    global fig
    # Plot all selected spectra
    ax_spec_cb.clear()
    for i, spectrum in enumerate(spectra_cb):
        ax_spec_cb.plot(wavelengths, spectrum, label=f'Selection {i+1}')

        # Find the peak wavelength for the first selection
        if i == 0:
            peak_index = np.argmax(spectrum)
            peak_wavelength = wavelengths[peak_index]
            color = get_color_from_wavelength(peak_wavelength)
            
            # Write peak wavelength in the top-right corner of the plot
            ax_spec_cb.text(
                0.95, 0.85,  # x, y position in axes coordinates (0.95, 0.95) corresponds to the top-right corner
                f'Peak: {peak_wavelength:.1f} nm\nColor: {color}\nSNR (full spectrum): {snr(selected_area_cb):.2f}\n SNR (channel: {current_channel}): {snr(selected_area_cb[current_channel]):.2f}',
                transform=ax_spec_cb.transAxes,  # Use axes coordinates for positioning
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                color='red'
            )

    ax_spec_cb.set_xlabel("Wavelength (nm)")
    ax_spec_cb.set_ylabel("Intensity")
    ax_spec_cb.set_title("Spectrum for Selected Areas")
    ax_spec_cb.legend()

    # Redraw the figure
    fig.canvas.draw_idle()

# Main program
def main():
    global ax_tl, ax_cb, ax_gen, ax_spec_cb, ax_spec_gen, fig, channel_slider
    global tl_image, cb_image, gen_image
    global colorbar_tl, colorbar_cb, colorbar_gen
    colorbar_tl, colorbar_cb, colorbar_gen = None, None, None
    
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

    # Create the plot
    fig = plt.figure(figsize=(16,10))

    # Add 3 equally sized subplots on the top row
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    ax_gen = fig.add_subplot(gs[0, 2])

    # Add 2 differently sized subplots on the bottom row
    ax_config = fig.add_subplot(gs[1, 0])  # This spans column 0
    ax_spec_cb = fig.add_subplot(gs[1, 1:3]) # This occupies column 1 -> 3

    fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.3, wspace=0.4)

    avrg_RASE, avrg_SRE = avrg_metrics()
    # Set window title
    fig.suptitle(f'Image Viewer (Average SRE for Test Dataset: {avrg_SRE:.3f})')
    fig.tight_layout(pad=5.0)

    # Initial plot
    tl_image = load_tl_image(current_tl_file)
    cb_image = load_cb_image(current_cb_file)
    gen_image = load_gen_image(current_gen_file)
    update_tl_plot(current_tl_file, current_pol)
    update_cb_plot(current_cb_file, current_channel)
    update_gen_plot(current_gen_file, current_channel)

    ## Widgets
    ax_config.axis('off')

    ax_id_textbox = plt.axes([0.10, 0.12, 0.2, 0.05])  # Adjusted position and size
    id_textbox = widgets.TextBox(ax_id_textbox, 'Jump to Image ID', initial=0)
    id_textbox.on_submit(change_img_id)

    ax_channel_slider = plt.axes([0.09, 0.22, 0.2, 0.03])
    channel_slider = widgets.Slider(
        ax=ax_channel_slider,
        label='Wavelength ',
        valmin=0,
        valmax=cb_image.shape[0] - 1,
        valinit=current_channel,
        valstep=1
    )
    channel_slider.on_changed(change_channel)

    # RectangleSelector for selecting region on the Cubert or Gen image
    rs1 = widgets.RectangleSelector(
        ax_cb,
        onselect,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )
    rs2 = widgets.RectangleSelector(
        ax_gen,
        onselect,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )

    ax_pol_slider = plt.axes([0.09, 0.28, 0.1, 0.03])
    pol_slider = widgets.Slider(
        ax=ax_pol_slider,
        label='Polarisation  ',
        valmin=0,
        valmax=135,
        valinit=current_pol,
        valstep=45
    )
    pol_slider.on_changed(change_pol)

    ax_next_button = plt.axes([0.135, 0.35, 0.1, 0.05])
    next_button = widgets.Button(ax_next_button, 'Next Image')
    next_button.on_clicked(next_image)

    ax_prev_button = plt.axes([0.03, 0.35, 0.1, 0.05])
    prev_button = widgets.Button(ax_prev_button, 'Previous Image')
    prev_button.on_clicked(prev_image)

    ax_clear_button = plt.axes([0.24, 0.35, 0.1, 0.05])
    clear_button = widgets.Button(ax_clear_button, 'Clear Spectra')
    clear_button.on_clicked(clear_spectra)

    # Display the plot
    plt.show()


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import os
from matplotlib import widgets

# Folder paths for images
thorlabs_image_folder = 'test'
cubert_image_folder = 'test'

# List available files in each folder
thorlabs_files = sorted([f for f in os.listdir(thorlabs_image_folder) if f.endswith(".tif") and f.startswith("tl_raw")])
cubert_files = sorted([f for f in os.listdir(cubert_image_folder) if f.endswith(".tif") and f.startswith("tl_gen")])

# Initial selections
current_img_index = 0
current_tl_file = thorlabs_files[current_img_index%len(thorlabs_files)]
current_cb_file = cubert_files[current_img_index%len(cubert_files)]
current_channel = 53
wavelengths = np.linspace(450, 850, 106)  # Assuming 106 channels from 450-850 nm

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

# Update the CB plot with the selected image and channel
def update_cb_plot(cb_file, channel):
    global ax_tl, fig  # Make sure cb_image is accessible in the callback
    global colorbar_cb
    global cb_image
    # Clear the current plot
    ax_cb.clear()
    # Plot selected channel of Cubert image
    img = cb_image[channel, :, :]
    im_cb = ax_cb.imshow(img, cmap='viridis')
    wavelength = wavelengths[channel]
    ax_cb.set_title(f"Generated Image: {cb_file} (Channel {channel}/{cb_image.shape[0]-1}, {wavelength:.1f} nm)")
    ax_cb.annotate(f"Channel Stats: \nSNR: {snr(img):.2f}, Min: {np.min(img):.2f}, Max: {np.max(img):.2f}, Avg: {np.mean(img):.2f}, Std: {np.std(img):.2f}", 
                   (-0.05,-0.18), xycoords='axes fraction')
    # create or update colorbars
    if colorbar_cb != None:
        colorbar_cb.update_normal(im_cb)
    else:
        colorbar_cb = plt.colorbar(im_cb)
    # Redraw the figure
    fig.canvas.draw_idle()

# Update the plot with the selected images and channel
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
    ax_tl.set_title(f"Thorlabs Image: {tl_file} (P={pol})")
    ax_tl.annotate(f"Channel Stats: \nSNR: {snr(img):.2f}, Min: {np.min(img):.2f}, Max: {np.max(img):.2f}, Avg: {np.mean(img):.2f}, Std: {np.std(img):.2f}", 
                   (-0.05,-0.18), xycoords='axes fraction')

    # create or update colorbars
    if colorbar_tl != None:
        colorbar_tl.update_normal(im_tl)
    else:
        colorbar_tl = plt.colorbar(im_tl)
    
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

# Change the channel
def change_channel(val):
    global current_channel
    current_channel = int(val)
    update_cb_plot(current_cb_file, current_channel)

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
    global cb_image, tl_image, current_img_index
    current_img_index += 1
    current_tl_file = thorlabs_files[current_img_index%len(thorlabs_files)]
    current_cb_file = cubert_files[current_img_index%len(cubert_files)]
    tl_image = load_tl_image(current_tl_file)
    cb_image = load_cb_image(current_cb_file)
    update_tl_plot(current_tl_file, current_pol)
    update_cb_plot(current_cb_file, current_channel)
    print(f"Showing next images: {current_tl_file} (TL), {current_cb_file} (CB)")

# Show prev image
def prev_image(_):
    global cb_image, tl_image, current_img_index
    current_img_index -= 1
    current_tl_file = thorlabs_files[current_img_index%len(thorlabs_files)]
    current_cb_file = cubert_files[current_img_index%len(cubert_files)]
    tl_image = load_tl_image(current_tl_file)
    cb_image = load_cb_image(current_cb_file)
    update_tl_plot(current_tl_file, current_pol)
    update_cb_plot(current_cb_file, current_channel)
    print(f"Showing previous images: {current_tl_file} (TL), {current_cb_file} (CB)")

# Calc SNR
def snr(img, axis=None, ddof=0):
    img = np.asanyarray(img)
    m = img.mean(axis)
    sd = img.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

# Callback function for region selection
def onselect(eclick, erelease):
    global selected_regions, cb_image, ax_intensity, fig

    # Get the coordinates of the rectangle
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    # Ensure coordinates are in the correct order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # Extract the reflectance values for the selected area
    selected_area = cb_image[:, y1:y2+1, x1:x2+1]

    # Calculate the average reflectance values for the selected area
    reflectance_values = np.mean(selected_area, axis=(1, 2))

    # Store the normalized reflectance values and color for this selection
    selected_regions.append(reflectance_values)

    # Plot all selected spectra
    ax_intensity.clear()
    for i, spectrum in enumerate(selected_regions):
        ax_intensity.plot(wavelengths, spectrum, label=f'Selection {i+1}')

        # Find the peak wavelength for the first selection
        if i == 0:
            peak_index = np.argmax(spectrum)
            peak_wavelength = wavelengths[peak_index]
            color = get_color_from_wavelength(peak_wavelength)
            
            # Write peak wavelength in the top-right corner of the plot
            ax_intensity.text(
                0.95, 0.95,  # x, y position in axes coordinates (0.95, 0.95) corresponds to the top-right corner
                f'Peak: {peak_wavelength:.1f} nm\nColor: {color}\nSNR (full spectrum): {snr(selected_area):.2f}\n SNR (channel: {current_channel}): {snr(selected_area[current_channel]):.2f}',
                transform=ax_intensity.transAxes,  # Use axes coordinates for positioning
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                color='red'
            )

    ax_intensity.set_xlabel("Wavelength (nm)")
    ax_intensity.set_ylabel("Intensity")
    ax_intensity.set_title("Spectrum for Selected Areas")
    ax_intensity.legend()

    # Redraw the figure
    fig.canvas.draw_idle()

# Main program
def main():
    global ax_tl, ax_cb, ax_intensity, fig, channel_slider
    global tl_image, cb_image
    global colorbar_tl, colorbar_cb
    colorbar_tl, colorbar_cb = None, None
    # Create the plot
    fig, (ax_tl, ax_cb, ax_intensity) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.3, wspace=0.4)

    # Set window title
    fig.suptitle('Paired Image Viewer')

    # Initial plot
    tl_image = load_tl_image(current_tl_file)
    cb_image = load_cb_image(current_cb_file)
    update_tl_plot(current_tl_file, current_pol)
    update_cb_plot(current_cb_file, current_channel)

    # Sliders and textboxes for selecting images and channel
    ax_tl_textbox = plt.axes([0.07, 0.12, 0.2, 0.05])  # Adjusted position and size
    tl_textbox = widgets.TextBox(ax_tl_textbox, 'Thorlabs File', initial=current_tl_file)
    tl_textbox.on_submit(change_thorlabs_file)

    ax_cb_textbox = plt.axes([0.4, 0.12, 0.2, 0.05])  # Adjusted position and size
    cb_textbox = widgets.TextBox(ax_cb_textbox, 'Cubert File', initial=current_cb_file)
    cb_textbox.on_submit(change_cubert_file)

    ax_channel_slider = plt.axes([0.4, 0.05, 0.2, 0.03])
    channel_slider = widgets.Slider(
        ax=ax_channel_slider,
        label='Wavelength',
        valmin=0,
        valmax=cb_image.shape[0] - 1,
        valinit=current_channel,
        valstep=1
    )
    channel_slider.on_changed(change_channel)

    # RectangleSelector for selecting region on the Cubert image
    rectangle_selector = widgets.RectangleSelector(
        ax_cb,
        onselect,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )

    ax_pol_slider = plt.axes([0.07, 0.05, 0.1, 0.03])
    pol_slider = widgets.Slider(
        ax=ax_pol_slider,
        label='Polarisation',
        valmin=0,
        valmax=135,
        valinit=current_pol,
        valstep=45
    )
    pol_slider.on_changed(change_pol)

    ax_next_button = plt.axes([0.85, 0.15, 0.1, 0.05])
    next_button = widgets.Button(ax_next_button, 'Next Image')
    next_button.on_clicked(next_image)

    ax_prev_button = plt.axes([0.7, 0.15, 0.1, 0.05])
    prev_button = widgets.Button(ax_prev_button, 'Previous Image')
    prev_button.on_clicked(prev_image)

    # Display the plot
    plt.show()


if __name__ == '__main__':
    main()
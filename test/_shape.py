from tifffile import imread

cb_raw = imread("test/cb_raw_0.tif")
tl_raw = imread("test/tl_raw_0.tif")
tl_gen = imread("test/tl_gen_0.tif")

print(cb_raw.shape)
print(tl_raw.shape)
print(tl_gen.shape)
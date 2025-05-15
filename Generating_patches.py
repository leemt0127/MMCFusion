from imageio import imread, imsave

# Modify the window size and stride to control the amount of final patches.
PATCH_SIZE = 384
PATCH_STRIDE = 80

# Original image pairs
num_image = 361

# Put the original images in the "IVIF_source/IR" & "IVIF_source/VIS" directories.
# The augmented data will be put in the "./IR" and "./VIS" directories.
prepath = "E:\\BackBone\\IV_patches"
patchesIR = []
patchesVIS = []
picidx = 0
for idx in range(0 + 1, num_image + 1):
    print("Decomposing " + str(idx) + "-th images...")
    imageIR = imread(prepath + '\\MSRS_dataset\\IR\\' + '{:03d}.png'.format(idx), pilmode='L')
    imageVIS = imread(prepath + '\\MSRS_dataset\\VIS\\' + '{:03d}.png'.format(idx), pilmode='L')
    h = imageIR.shape[0]
    w = imageIR.shape[1]
    for i in range(0, h - PATCH_SIZE + 1, PATCH_STRIDE):
        for j in range(0, w - PATCH_SIZE + 1, PATCH_STRIDE):
            picidx += 1
            patchImageIR = imageIR[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
            patchImageVIS = imageVIS[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
            imsave(prepath + '/MSRS_train_384/IR/' + str(picidx) + '.png', patchImageIR)
            imsave(prepath + '/MSRS_train_384/VIS/' + str(picidx) + '.png', patchImageVIS)
    print(picidx)

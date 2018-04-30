import SimpleITK as sitk
def nii_save(img,path):
	sitk_img = sitk.GetImageFromArray(np.transpose(img[:,:,:,np.newaxis],(0,1,2,3)), isVector=True)
	sitk.WriteImage(sitk_img,path)

for i in range(kk.shape[0]):
    nii_save(kk[i],'D:/Inspiration/OU/2-ANNA/project/result_image/case'+str(i)+'img.nii')
    nii_save(gg[i,:,:,:,0],'D:/Inspiration/OU/2-ANNA/project/result_image/case'+str(i)+'mask.nii')
    nii_save(jj[i,:,:,:],'D:/Inspiration/OU/2-ANNA/project/result_image/case'+str(i)+'truth.nii')

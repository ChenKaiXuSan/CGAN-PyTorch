from cleanfid import fid 

fake_img = "samples/1104_cifar10_10kepochs_hatano/9900/fake_images/"
real_img = "samples/1104_cifar10_10kepochs_hatano/9900/real_images"

# score = fid.compute_fid(fake_img, real_img)

# print(score)


fid_score = fid.compute_fid(fake_img, dataset_name="cifar10", dataset_res=32,  mode="clean", dataset_split="train")

print(fid_score)

fid_score = fid.compute_fid(fake_img, real_img)

print(fid_score)

score = fid.compute_kid(fake_img, real_img)

print(score)
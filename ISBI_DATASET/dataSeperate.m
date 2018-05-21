dirRoot = '/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation/';
mode = 'originalEnColor/';
mode1 = 'fusedImage/';
mode2 ='/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation/trainGroundTruth/';
if ~exist(mode2)
    mkdir(mode2);
else
    delete([mode2 '*'])
end
mode3 = '/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation/trainInput/';
if ~exist(mode3)
    mkdir(mode3);
else
    delete([mode3 '*'])
end
mode4 ='/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation/valiGroundTruth/';
if ~exist(mode4)
    mkdir(mode4);
else
    delete([mode4 '*'])
end
mode5 ='/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation/valiInput/';
if ~exist(mode5)
    mkdir(mode5);
else
    delete([mode5 '*'])
end
f_dir1 = dir([dirRoot, mode]);
f_dir = dir([dirRoot,mode1]);
for f = 3:length(f_dir)
    fileName = f_dir(f).name;
    if strcmp(fileName(end-2:end),'tif')
        string = regexp(fileName,'_','split');
        idx = str2num(string{2});
        idx
        trainMask = [dirRoot, mode1, fileName];
        trainColor= [dirRoot, mode,fileName(1:9),fileName(end-7:end-4),'.jpg'];
        mask = imread(trainMask);
        color = imread(trainColor);
        if idx<=4
            imwrite(mask,[mode4,fileName]);
            imwrite(color,[mode5,fileName(1:9),fileName(end-7:end-4),'.jpg']);
        else
            imwrite(mask,[mode2,fileName]);
            imwrite(color,[mode3,fileName(1:9),fileName(end-7:end-4),'.jpg']);
        end
    end
end

dirRoot = '/home/mpl/medical_image_classification/ISBI_DATASET/Lesion_Segmentation/';
% system(['python /home/mpl/medical_image_classification/src/colorEnchance.py']);
mode = 'Apparent_Retinopathy/'
% mode1 = 'fusedImage/';
if ~exist([dirRoot,'fusedImage/'])
    mkdir([dirRoot,'fusedImage/']);
else
    delete([dirRoot,'fusedImage/', '*'])
end

if ~exist([dirRoot,'original_au/'])
    mkdir([dirRoot,'original_au/']);
else
    delete([dirRoot,'original_au/', '*'])
end

f_dir = dir([dirRoot, mode]);
resolution = [1024,1024];
for f = 3:length(f_dir)
    fileName = f_dir(f).name;
    if strcmp(fileName(end-2:end),'jpg')
        fileRoot = [dirRoot, mode, f_dir(f).name];
        img = imread(fileRoot);
        
        [img,pbbox]=imAutoResize(img);
        img1 = imresize( img,resolution );
        y=1;
        x=randperm(100,1);
        m= size(img,1);
        n= size(img,2);
        img11 = imresize(imcrop(img, [y,x,n-2*y,m-2*x]),resolution);
        c = randperm(40,1)+5;
        img(:,:,3)= img(:,:,3) +c;
        img12 = imresize(img,resolution);
        %-----------------------------------------------
        mode1 = 'original_au/';
        imwrite(img1,[dirRoot,mode1,strrep(fileName, '.jpg','_or01.jpg')]);
        imwrite(imrotate(img1,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or02.jpg')]);
        imwrite(imrotate(img1,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or03.jpg')]);
        imwrite(imrotate(img1,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or04.jpg')]);
        
        imwrite(img11,[dirRoot,mode1,strrep(fileName, '.jpg','_or05.jpg')]);
        imwrite(imrotate(img11,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or06.jpg')]);
        imwrite(imrotate(img11,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or07.jpg')]);
        imwrite(imrotate(img11,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or08.jpg')]);
        
        imwrite(img12,[dirRoot,mode1,strrep(fileName, '.jpg','_or09.jpg')]);
        imwrite(imrotate(img12,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or10.jpg')]);
        imwrite(imrotate(img12,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or11.jpg')]);
        imwrite(imrotate(img12,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or12.jpg')]);
        %----------------------------------------------
        imwrite(img1(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr01.jpg')]);
        imwrite(imrotate(img1(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr02.jpg')]);
        imwrite(imrotate(img1(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr03.jpg')]);
        imwrite(imrotate(img1(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr04.jpg')]);
        
        imwrite(img11(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr05.jpg')]);
        imwrite(imrotate(img11(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr06.jpg')]);
        imwrite(imrotate(img11(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr07.jpg')]);
        imwrite(imrotate(img11(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr08.jpg')]);
        
        imwrite(img12(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr09.jpg')]);
        imwrite(imrotate(img12(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr10.jpg')]);
        imwrite(imrotate(img12(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr11.jpg')]);
        imwrite(imrotate(img12(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr12.jpg')]);

        %------------------------------------------------
        imwrite(img1(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud01.jpg')]);
        imwrite(imrotate(img1(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud02.jpg')]);
        imwrite(imrotate(img1(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud03.jpg')]);
        imwrite(imrotate(img1(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud04.jpg')]);
        
        imwrite(img11(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud05.jpg')]);
        imwrite(imrotate(img11(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud06.jpg')]);
        imwrite(imrotate(img11(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud07.jpg')]);
        imwrite(imrotate(img11(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud08.jpg')]);
        
        imwrite(img12(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud09.jpg')]);
        imwrite(imrotate(img12(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud10.jpg')]);
        imwrite(imrotate(img12(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud11.jpg')]);
        imwrite(imrotate(img12(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud12.jpg')]);

        %-----------------------------------------------
        fileName1 = strrep(fileName, '.jpg','_EX.tif')
        mode1 = 'EX/';
        fileRoot =[dirRoot, mode1, fileName1];
        if exist(fileRoot)
            img = imread(fileRoot);
            img = img(pbbox(1,2):pbbox(3,2),pbbox(1,1):pbbox(3,1));
            img2 = imresize(img,resolution);
%             imresize(imcrop(img, [y,x,n-2*y,m-2*x]),resolution)
            img21 = imresize(imcrop(img, [y,x,n-2*y,m-2*x]),resolution);
            img22 = img2;
            %---------------------------------------
%             mode1 = 'EX_au/';
%             img1=img2;
%             img11 = img21;
%             img12 =img22;
%             imwrite(img1,[dirRoot,mode1,strrep(fileName, '.jpg','_or01.jpg')]);
%             imwrite(imrotate(img1,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or02.jpg')]);
%             imwrite(imrotate(img1,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or03.jpg')]);
%             imwrite(imrotate(img1,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or04.jpg')]);
% 
%             imwrite(img11,[dirRoot,mode1,strrep(fileName, '.jpg','_or05.jpg')]);
%             imwrite(imrotate(img11,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or06.jpg')]);
%             imwrite(imrotate(img11,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or07.jpg')]);
%             imwrite(imrotate(img11,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or08.jpg')]);
% 
%             imwrite(img12,[dirRoot,mode1,strrep(fileName, '.jpg','_or09.jpg')]);
%             imwrite(imrotate(img12,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or10.jpg')]);
%             imwrite(imrotate(img12,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or11.jpg')]);
%             imwrite(imrotate(img12,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or12.jpg')]);
%             %----------------------------------------------
%             imwrite(img1(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr01.jpg')]);
%             imwrite(imrotate(img1(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr02.jpg')]);
%             imwrite(imrotate(img1(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr03.jpg')]);
%             imwrite(imrotate(img1(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr04.jpg')]);
% 
%             imwrite(img11(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr05.jpg')]);
%             imwrite(imrotate(img11(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr06.jpg')]);
%             imwrite(imrotate(img11(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr07.jpg')]);
%             imwrite(imrotate(img11(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr08.jpg')]);
% 
%             imwrite(img12(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr09.jpg')]);
%             imwrite(imrotate(img12(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr10.jpg')]);
%             imwrite(imrotate(img12(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr11.jpg')]);
%             imwrite(imrotate(img12(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr12.jpg')]);
% 
%             %------------------------------------------------
%             imwrite(img1(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud01.jpg')]);
%             imwrite(imrotate(img1(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud02.jpg')]);
%             imwrite(imrotate(img1(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud03.jpg')]);
%             imwrite(imrotate(img1(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud04.jpg')]);
% 
%             imwrite(img11(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud05.jpg')]);
%             imwrite(imrotate(img11(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud06.jpg')]);
%             imwrite(imrotate(img11(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud07.jpg')]);
%             imwrite(imrotate(img11(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud08.jpg')]);
% 
%             imwrite(img12(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud09.jpg')]);
%             imwrite(imrotate(img12(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud10.jpg')]);
%             imwrite(imrotate(img12(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud11.jpg')]);
%             imwrite(imrotate(img12(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud12.jpg')]);
            %-------------------------------------
            fileName1 = strrep(fileName, '.jpg','_HE.tif');
            mode1 = 'HE/';
            fileRoot =[dirRoot, mode1, fileName1];
            if exist(fileRoot)
                img = imread(fileRoot);
                img = img(pbbox(1,2):pbbox(3,2),pbbox(1,1):pbbox(3,1));
                img3 = imresize(img,resolution);
                img31 = imresize(imcrop(img, [y,x,n-2*y,m-2*x]),resolution);
                img32 = img3;
%                 mode1 = 'HE_au/';
%             img1=img3;
%             img11 = img31;
%             img12 =img32;
%             imwrite(img1,[dirRoot,mode1,strrep(fileName, '.jpg','_or01.jpg')]);
%             imwrite(imrotate(img1,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or02.jpg')]);
%             imwrite(imrotate(img1,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or03.jpg')]);
%             imwrite(imrotate(img1,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or04.jpg')]);
% 
%             imwrite(img11,[dirRoot,mode1,strrep(fileName, '.jpg','_or05.jpg')]);
%             imwrite(imrotate(img11,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or06.jpg')]);
%             imwrite(imrotate(img11,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or07.jpg')]);
%             imwrite(imrotate(img11,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or08.jpg')]);
% 
%             imwrite(img12,[dirRoot,mode1,strrep(fileName, '.jpg','_or09.jpg')]);
%             imwrite(imrotate(img12,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or10.jpg')]);
%             imwrite(imrotate(img12,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or11.jpg')]);
%             imwrite(imrotate(img12,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or12.jpg')]);
%             %----------------------------------------------
%             imwrite(img1(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr01.jpg')]);
%             imwrite(imrotate(img1(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr02.jpg')]);
%             imwrite(imrotate(img1(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr03.jpg')]);
%             imwrite(imrotate(img1(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr04.jpg')]);
% 
%             imwrite(img11(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr05.jpg')]);
%             imwrite(imrotate(img11(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr06.jpg')]);
%             imwrite(imrotate(img11(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr07.jpg')]);
%             imwrite(imrotate(img11(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr08.jpg')]);
% 
%             imwrite(img12(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr09.jpg')]);
%             imwrite(imrotate(img12(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr10.jpg')]);
%             imwrite(imrotate(img12(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr11.jpg')]);
%             imwrite(imrotate(img12(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr12.jpg')]);
% 
%             %------------------------------------------------
%             imwrite(img1(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud01.jpg')]);
%             imwrite(imrotate(img1(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud02.jpg')]);
%             imwrite(imrotate(img1(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud03.jpg')]);
%             imwrite(imrotate(img1(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud04.jpg')]);
% 
%             imwrite(img11(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud05.jpg')]);
%             imwrite(imrotate(img11(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud06.jpg')]);
%             imwrite(imrotate(img11(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud07.jpg')]);
%             imwrite(imrotate(img11(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud08.jpg')]);
% 
%             imwrite(img12(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud09.jpg')]);
%             imwrite(imrotate(img12(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud10.jpg')]);
%             imwrite(imrotate(img12(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud11.jpg')]);
%             imwrite(imrotate(img12(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud12.jpg')]);
                fileName1 = strrep(fileName, '.jpg','_MA.tif')
                mode1 = 'MA/';
                fileRoot =[dirRoot, mode1, fileName1];
                if exist(fileRoot)
                    img = imread(fileRoot);
                    img = img(pbbox(1,2):pbbox(3,2),pbbox(1,1):pbbox(3,1));
                    img4 = imresize(img,resolution);
                    img41 = imresize(imcrop(img, [y,x,n-2*y,m-2*x]),resolution);
                    img42 = img4;
                    %---------------------
%                     mode1 = 'MA_au/';
%             img1=img4;
%             img11 = img41;
%             img12 =img42;
%             imwrite(img1,[dirRoot,mode1,strrep(fileName, '.jpg','_or01.jpg')]);
%             imwrite(imrotate(img1,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or02.jpg')]);
%             imwrite(imrotate(img1,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or03.jpg')]);
%             imwrite(imrotate(img1,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or04.jpg')]);
% 
%             imwrite(img11,[dirRoot,mode1,strrep(fileName, '.jpg','_or05.jpg')]);
%             imwrite(imrotate(img11,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or06.jpg')]);
%             imwrite(imrotate(img11,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or07.jpg')]);
%             imwrite(imrotate(img11,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or08.jpg')]);
% 
%             imwrite(img12,[dirRoot,mode1,strrep(fileName, '.jpg','_or09.jpg')]);
%             imwrite(imrotate(img12,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or10.jpg')]);
%             imwrite(imrotate(img12,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or11.jpg')]);
%             imwrite(imrotate(img12,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or12.jpg')]);
%             %----------------------------------------------
%             imwrite(img1(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr01.jpg')]);
%             imwrite(imrotate(img1(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr02.jpg')]);
%             imwrite(imrotate(img1(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr03.jpg')]);
%             imwrite(imrotate(img1(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr04.jpg')]);
% 
%             imwrite(img11(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr05.jpg')]);
%             imwrite(imrotate(img11(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr06.jpg')]);
%             imwrite(imrotate(img11(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr07.jpg')]);
%             imwrite(imrotate(img11(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr08.jpg')]);
% 
%             imwrite(img12(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr09.jpg')]);
%             imwrite(imrotate(img12(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr10.jpg')]);
%             imwrite(imrotate(img12(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr11.jpg')]);
%             imwrite(imrotate(img12(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr12.jpg')]);
% 
%             %------------------------------------------------
%             imwrite(img1(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud01.jpg')]);
%             imwrite(imrotate(img1(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud02.jpg')]);
%             imwrite(imrotate(img1(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud03.jpg')]);
%             imwrite(imrotate(img1(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud04.jpg')]);
% 
%             imwrite(img11(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud05.jpg')]);
%             imwrite(imrotate(img11(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud06.jpg')]);
%             imwrite(imrotate(img11(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud07.jpg')]);
%             imwrite(imrotate(img11(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud08.jpg')]);
% 
%             imwrite(img12(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud09.jpg')]);
%             imwrite(imrotate(img12(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud10.jpg')]);
%             imwrite(imrotate(img12(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud11.jpg')]);
%             imwrite(imrotate(img12(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud12.jpg')]);
%                     %------------------------------------
                    fileName1 = strrep(fileName, '.jpg','_SE.tif');
                    mode1 = 'SE/';
                    fileRoot =[dirRoot, mode1, fileName1];
                    if exist(fileRoot)
                        img = imread(fileRoot);
                        img = img(pbbox(1,2):pbbox(3,2),pbbox(1,1):pbbox(3,1));
                        img5 = imresize(img,resolution);
                        img51 = imresize(imcrop(img, [y,x,n-2*y,m-2*x]),resolution);
                        img52 = img5;
%                         %------------------------
%                         mode1 = 'SE_au/';
%             img1=img5;
%             img11 = img51;
%             img12 =img52;
%             imwrite(img1,[dirRoot,mode1,strrep(fileName, '.jpg','_or01.jpg')]);
%             imwrite(imrotate(img1,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or02.jpg')]);
%             imwrite(imrotate(img1,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or03.jpg')]);
%             imwrite(imrotate(img1,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or04.jpg')]);
% 
%             imwrite(img11,[dirRoot,mode1,strrep(fileName, '.jpg','_or05.jpg')]);
%             imwrite(imrotate(img11,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or06.jpg')]);
%             imwrite(imrotate(img11,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or07.jpg')]);
%             imwrite(imrotate(img11,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or08.jpg')]);
% 
%             imwrite(img12,[dirRoot,mode1,strrep(fileName, '.jpg','_or09.jpg')]);
%             imwrite(imrotate(img12,90),[dirRoot,mode1,strrep(fileName, '.jpg','_or10.jpg')]);
%             imwrite(imrotate(img12,180),[dirRoot,mode1,strrep(fileName, '.jpg','_or11.jpg')]);
%             imwrite(imrotate(img12,270),[dirRoot,mode1,strrep(fileName, '.jpg','_or12.jpg')]);
%             %----------------------------------------------
%             imwrite(img1(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr01.jpg')]);
%             imwrite(imrotate(img1(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr02.jpg')]);
%             imwrite(imrotate(img1(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr03.jpg')]);
%             imwrite(imrotate(img1(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr04.jpg')]);
% 
%             imwrite(img11(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr05.jpg')]);
%             imwrite(imrotate(img11(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr06.jpg')]);
%             imwrite(imrotate(img11(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr07.jpg')]);
%             imwrite(imrotate(img11(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr08.jpg')]);
% 
%             imwrite(img12(:,end:-1:1,:),[dirRoot,mode1,strrep(fileName, '.jpg','_lr09.jpg')]);
%             imwrite(imrotate(img12(:,end:-1:1,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_lr10.jpg')]);
%             imwrite(imrotate(img12(:,end:-1:1,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_lr11.jpg')]);
%             imwrite(imrotate(img12(:,end:-1:1,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_lr12.jpg')]);
% 
%             %------------------------------------------------
%             imwrite(img1(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud01.jpg')]);
%             imwrite(imrotate(img1(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud02.jpg')]);
%             imwrite(imrotate(img1(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud03.jpg')]);
%             imwrite(imrotate(img1(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud04.jpg')]);
% 
%             imwrite(img11(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud05.jpg')]);
%             imwrite(imrotate(img11(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud06.jpg')]);
%             imwrite(imrotate(img11(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud07.jpg')]);
%             imwrite(imrotate(img11(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud08.jpg')]);
% 
%             imwrite(img12(end:-1:1,:,:),[dirRoot,mode1,strrep(fileName, '.jpg','_ud09.jpg')]);
%             imwrite(imrotate(img12(end:-1:1,:,:),90),[dirRoot,mode1,strrep(fileName, '.jpg','_ud10.jpg')]);
%             imwrite(imrotate(img12(end:-1:1,:,:),180),[dirRoot,mode1,strrep(fileName, '.jpg','_ud11.jpg')]);
%             imwrite(imrotate(img12(end:-1:1,:,:),270),[dirRoot,mode1,strrep(fileName, '.jpg','_ud12.jpg')]);
                        
                        
                        temp =   img2|img3|img4|img5;
                        temp2 = img21|img31|img41|img51;
                        temp3 = img22|img32|img42|img52;

                        imwrite(temp,[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_or01.tif')]);
                        imwrite(imrotate(temp,90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_or02.tif')]);
                        imwrite(imrotate(temp,180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_or03.tif')]);
                        imwrite(imrotate(temp,270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_or04.tif')]);
                        
                        imwrite(temp2,[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_or05.tif')]);
                        imwrite(imrotate(temp2,90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_or06.tif')]);
                        imwrite(imrotate(temp2,180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_or07.tif')]);
                        imwrite(imrotate(temp2,270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_or08.tif')]);
                        
                        imwrite(temp3,[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_or09.tif')]);
                        imwrite(imrotate(temp3,90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_or10.tif')]);
                        imwrite(imrotate(temp3,180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_or11.tif')]);
                        imwrite(imrotate(temp3,270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_or12.tif')]);
                        %---------------------------------
                        imwrite(temp(:,end:-1:1),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_lr01.tif')]);
                        imwrite(imrotate(temp(:,end:-1:1),90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_lr02.tif')]);
                        imwrite(imrotate(temp(:,end:-1:1),180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_lr03.tif')]);
                        imwrite(imrotate(temp(:,end:-1:1),270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_lr04.tif')]);
                        
                        imwrite(temp2(:,end:-1:1),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_lr05.tif')]);
                        imwrite(imrotate(temp2(:,end:-1:1),90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_lr06.tif')]);
                        imwrite(imrotate(temp2(:,end:-1:1),180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_lr07.tif')]);
                        imwrite(imrotate(temp2(:,end:-1:1),270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_lr08.tif')]);
                        
                        imwrite(temp3(:,end:-1:1),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_lr09.tif')]);
                        imwrite(imrotate(temp3(:,end:-1:1),90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_lr10.tif')]);
                        imwrite(imrotate(temp3(:,end:-1:1),180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_lr11.tif')]);
                        imwrite(imrotate(temp3(:,end:-1:1),270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_lr12.tif')]);
                        %-------------------------------
                        imwrite(temp(end:-1:1,:),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_ud01.tif')]);
                        imwrite(imrotate(temp(end:-1:1,:),90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_ud02.tif')]);
                        imwrite(imrotate(temp(end:-1:1,:),180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_ud03.tif')]);
                        imwrite(imrotate(temp(end:-1:1,:),270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_ud04.tif')]);
                        
                        imwrite(temp2(end:-1:1,:),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_ud05.tif')]);
                        imwrite(imrotate(temp2(end:-1:1,:),90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_ud06.tif')]);
                        imwrite(imrotate(temp2(end:-1:1,:),180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_ud07.tif')]);
                        imwrite(imrotate(temp2(end:-1:1,:),270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_ud08.tif')]);
                        
                        imwrite(temp3(end:-1:1,:),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_ud09.tif')]);
                        imwrite(imrotate(temp3(end:-1:1,:),90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_ud10.tif')]);
                        imwrite(imrotate(temp3(end:-1:1,:),180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_ud11.tif')]);
                        imwrite(imrotate(temp3(end:-1:1,:),270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_SE_ud12.tif')]);
                        %-----------------------------
                       
                    else
                        temp =img2|img4|img3;
                        temp2 =img21|img41|img31;
                        temp3 =img22|img42|img32;
                        
                        imwrite(temp,[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_or01.tif')]);
                        imwrite(imrotate(temp,90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_or02.tif')]);
                        imwrite(imrotate(temp,180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_or03.tif')]);
                        imwrite(imrotate(temp,270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_or04.tif')]);
                        
                        imwrite(temp2,[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_or05.tif')]);
                        imwrite(imrotate(temp2,90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_or06.tif')]);
                        imwrite(imrotate(temp2,180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_or07.tif')]);
                        imwrite(imrotate(temp2,270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_or08.tif')]);
                        
                        imwrite(temp3,[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_or09.tif')]);
                        imwrite(imrotate(temp3,90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_or10.tif')]);
                        imwrite(imrotate(temp3,180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_or11.tif')]);
                        imwrite(imrotate(temp3,270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_or12.tif')]);
                        %---------------------------------
                        imwrite(temp(:,end:-1:1),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_lr01.tif')]);
                        imwrite(imrotate(temp(:,end:-1:1),90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_lr02.tif')]);
                        imwrite(imrotate(temp(:,end:-1:1),180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_lr03.tif')]);
                        imwrite(imrotate(temp(:,end:-1:1),270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_lr04.tif')]);
                        
                        imwrite(temp2(:,end:-1:1),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_lr05.tif')]);
                        imwrite(imrotate(temp2(:,end:-1:1),90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_lr06.tif')]);
                        imwrite(imrotate(temp2(:,end:-1:1),180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_lr07.tif')]);
                        imwrite(imrotate(temp2(:,end:-1:1),270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_lr08.tif')]);
                        
                        imwrite(temp3(:,end:-1:1),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_lr09.tif')]);
                        imwrite(imrotate(temp3(:,end:-1:1),90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_lr10.tif')]);
                        imwrite(imrotate(temp3(:,end:-1:1),180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_lr11.tif')]);
                        imwrite(imrotate(temp3(:,end:-1:1),270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_lr12.tif')]);
                        %-------------------------------
                        imwrite(temp(end:-1:1,:),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_ud01.tif')]);
                        imwrite(imrotate(temp(end:-1:1,:),90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_ud02.tif')]);
                        imwrite(imrotate(temp(end:-1:1,:),180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_ud03.tif')]);
                        imwrite(imrotate(temp(end:-1:1,:),270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_ud04.tif')]);
                        
                        imwrite(temp2(end:-1:1,:),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_ud05.tif')]);
                        imwrite(imrotate(temp2(end:-1:1,:),90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_ud06.tif')]);
                        imwrite(imrotate(temp2(end:-1:1,:),180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_ud07.tif')]);
                        imwrite(imrotate(temp2(end:-1:1,:),270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_ud08.tif')]);
                        
                        imwrite(temp3(end:-1:1,:),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_ud09.tif')]);
                        imwrite(imrotate(temp3(end:-1:1,:),90),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_ud10.tif')]);
                        imwrite(imrotate(temp3(end:-1:1,:),180),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_ud11.tif')]);
                        imwrite(imrotate(temp3(end:-1:1,:),270),[dirRoot,'fusedImage/',strrep(fileName, '.jpg','_EX_HE_MA_ud12.tif')]);
                        %-----------------------------


                    end
                end
            end
        end
    end

end  



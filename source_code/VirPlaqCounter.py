import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit, QFileDialog, QMessageBox
from PyQt5 import uic


def choose_image_file_names(input_dir: str)->list:
    """
    Take list of files and directories in input_dir and choose names, which are images ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

    Parameters
    ----------
    input_dir : str
        input directory with images

    Returns
    -------
    list_with_images_names : list
        list with strings, which are names of .jpg files in input_dir.

    """
    list_with_everything = os.listdir(input_dir)
    list_with_images_names = []
    for name in list_with_everything:
        if name.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            list_with_images_names.append(name)
    if not list_with_images_names:
        raise ValueError('ValueError 33: Could not find images in %s' % (input_dir))
    return list_with_images_names


def cluster_img(initial_img: np.array)->np.array:
    """
    Cluster initial image with cv2.KMEANS_RANDOM_CENTERS

    Parameters
    ----------
    initial_img : np.array
        initial image

    Returns
    -------
    clustered_img : TYPE
        clustared image in black and white

    """
    img_for_processing = initial_img.reshape((-1, 3))
    img_for_processing = np.float32(img_for_processing)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    number_of_clusters = 4
    compactness, labels, centers = cv2.kmeans(img_for_processing, number_of_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    flatten_clustered_img = centers[labels.flatten()]
    clustered_img = flatten_clustered_img.reshape((initial_img.shape))
    clustered_img = cv2.cvtColor(clustered_img, cv2.COLOR_RGB2GRAY)
    return clustered_img


def mask_img(clustered_img: np.array, initial_img: np.array)->np.array:
    """
    Mask the original image so that only the holes remain

    Parameters
    ----------
    clustered_img : np.array
        image after cluster_img function (clustered gray img)
    initial_img : np.array
        initial image

    Returns
    -------
    masked : np.array
        initial image after masking: only the holes remain in the image, the rest turns black
    mask_img : np.array
        image of mask (only holes)

    """
    rows, cols = clustered_img.shape
    darkest_color = np.min(clustered_img)
    not_darkest_mask = np.where(clustered_img != darkest_color, 0, 1)
    not_darkest_mask = (not_darkest_mask * 255).astype(np.uint8)
    kernel_k = 8
    kernel = np.ones((kernel_k, kernel_k), np.float32)/25
    raw_mask = cv2.filter2D(not_darkest_mask, -1, kernel)
    mask = (np.where(raw_mask < 128, 0, 1)).astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    masked = initial_img * mask
    mask_img = (mask * 255).astype(np.uint8)
    return masked, mask_img


def detect_circles(mask_img: np.array, r_min: int, r_max: int)->np.array:
    """
    Detect parameters of circles (holes) in mask 

    Parameters
    ----------
    mask_img : np.array
        image of mask with holes
    r_min : int
        min radius for cv2.HoughCircles
    r_max : int
        max radius for cv2.HoughCircles

    Returns
    -------
    circles_params : np.array
        parametres of detected circles: x_center, y_center, radius

    """
    rows = mask_img.shape[0]
    circles_params = cv2.HoughCircles(mask_img, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1 = 100, param2 = 30,
                               minRadius = r_min, maxRadius = r_max)
    return circles_params


def draw_circles(mask: np.array, circles_params: np.array)->np.array:
    """
    Draw circles on the image of mask for controlling this step

    Parameters
    ----------
    mask : np.array
        image of mask with holes
    circles_params : np.array
        parametres of detected circles: x_center, y_center, radius

    Returns
    -------
    resized_img : np.array
        image of mask with drawn circles on it

    """
    drawing_img = cv2.merge((mask, mask, mask))
    drawing_params = np.round(circles_params[0, :]).astype("int")
    for (x, y, r) in drawing_params:    
        cv2.circle(drawing_img, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(drawing_img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    scale_percent = 60
    width = int(drawing_img.shape[1] * scale_percent / 100)
    height = int(drawing_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(drawing_img, dim, interpolation = cv2.INTER_AREA)
    return resized_img



def save_mask_with_circles_(mask: np.array, circles_params: np.array, path_out: str, img_name: str):
    """
    Draw and then save circles on the image of mask for controlling this step

    Parameters
    ----------
    mask : np.array
        image of mask with holes
    circles_params : np.array
        parametres of detected circles: x_center, y_center, radius
    path_out : str
        path to the directory for saving
    img_name : str
        name of initial image

    Returns
    -------
    None.

    """
    out_name = os.path.join(path_out, ('maskimg_' + img_name + '.jpg'))
    resized_img = draw_circles(mask, circles_params)
    cv2.imwrite(out_name, resized_img)


def visualise_circle(mask, circles_params):
    resized_img = draw_circles(mask, circles_params)
    cv2.imshow('Check circles', resized_img)
    cv2.waitKey(0)


def cluster_gr(imag: np.array, number_of_clusters: int)->np.array:
    """
    Cluster image in black and white colors with cv2.KMEANS_RANDOM_CENTERS in calc_obj finction

    Parameters
    ----------
    imag : np.array
        DESCRIPTION.
    number_of_clusters : int
        number of clusters 

    Returns
    -------
    wclust : TYPE
        clustared image

    """
    img_for_processing = imag.reshape((-1,1))
    img_for_processing = np.float32(img_for_processing)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(img_for_processing, number_of_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    flatten_clustered_img = centers[labels.flatten()]
    clustered_img = flatten_clustered_img.reshape((imag.shape))
    return clustered_img






def dictMaker(Crugi):
    i=0
    ty=np.zeros((0,3))
    for i in range(0, 24):
        temp_x = Crugi[0][i][0]
        temp_y = Crugi[0][i][1]
        temp_R = Crugi[0][i][2]
        temp = [(temp_x,temp_y,temp_R)]
        ty = np.append(ty, temp, axis=0)
    rt=ty[ty[:, 0].argsort()]
    points = dict()
    i=0
    for i in range(0,24):
        if i == 0:
            t=0
            temp = rt[i:i+6,0:3]
            temp2=temp[temp[:, 1].argsort()]
            points.update(D6=temp2[t:t+1,:])
            points.update(D5=temp2[t+1:t+2,:])
            points.update(D4=temp2[t+2:t+3,:])
            points.update(D3=temp2[t+3:t+4,:])
            points.update(D2=temp2[t+4:t+5,:])
            points.update(D1=temp2[t+5:t+6,:])
        if i == 6:
            t=0
            temp = rt[i:i+6,0:3]
            temp2=temp[temp[:, 1].argsort()]
            points.update(C6=temp2[t:t+1,:])
            points.update(C5=temp2[t+1:t+2,:])
            points.update(C4=temp2[t+2:t+3,:])
            points.update(C3=temp2[t+3:t+4,:])
            points.update(C2=temp2[t+4:t+5,:])
            points.update(C1=temp2[t+5:t+6,:])
        if i == 12:
            t=0
            temp = rt[i:i+6,0:3]
            temp2=temp[temp[:, 1].argsort()]
            points.update(B6=temp2[t:t+1,:])
            points.update(B5=temp2[t+1:t+2,:])
            points.update(B4=temp2[t+2:t+3,:])
            points.update(B3=temp2[t+3:t+4,:])
            points.update(B2=temp2[t+4:t+5,:])
            points.update(B1=temp2[t+5:t+6,:])
        if i == 18:
            t=0
            temp = rt[i:i+6,0:3]
            temp2=temp[temp[:, 1].argsort()]
            points.update(A6=temp2[t:t+1,:])
            points.update(A5=temp2[t+1:t+2,:])
            points.update(A4=temp2[t+2:t+3,:])
            points.update(A3=temp2[t+3:t+4,:])
            points.update(A2=temp2[t+4:t+5,:])
            points.update(A1=temp2[t+5:t+6,:])
    return points

def Bbox(x, y, R):
    y1 = y - R
    x1 = x - R
    y2 = y + R
    x2 = x + R
    return y1, x1, y2, x2

def del_circ(cut_clust):
    R = (cut_clust.shape[0])/2
    b = (np.unique(cut_clust))
    dn_min = int(np.min(b))
    dn_max = np.max(b)
    c=int(cut_clust.shape[0]/2)
    color = (dn_min, dn_min, dn_min)
    res=cv2.circle(cut_clust,(c,c), (c), color=color, thickness = 9)
    return res


def calc_obj(points, masked, path_out):
    result = dict()
    for key, value in points.items():
        k=key
        p = value
        y1, x1, y2, x2 = Bbox(p[0][0], p[0][1], p[0][2])
        y1=int(round(y1))
        x1=int(round(x1))
        y2=int(round(y2))
        x2=int(round(x2))
        cut = masked[y1:y2,x1:x2]
        nt = path_out + 'cut_' + key + '.jpg'
        cv2.imwrite(nt, cut)
        cut = cv2.cvtColor(cut, cv2.COLOR_RGB2GRAY)
        cut = cluster_gr(cut,2)
        imgray = cut
        imgray = del_circ(imgray)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new=cv2.merge((imgray,imgray,imgray))
        cont = cv2.drawContours(new, contours, -1, (0,255,0), 1)
        ns = path_out + 'cont_' + key + '.jpg' 
        cv2.imwrite(ns, cont)
        if hierarchy is not None:
            count = int(hierarchy.shape[1])
            result[k]=count
        else:
            result[k]=0
    print('RES',result)
    return result



def save_to_one_excel(tablets_results: dict, output_dir: str):
    """
    Save counted plaques in tablet (on the image) into separate excel files: each sheet for each image.

    Parameters
    ----------
    tablet_results : dict
        dictionary where the key is the name of the image file, and the value is a dictionary where the keys are the cells of the tablet, the values are the number of plaques
    output_dir : str
        directory for saving the excel file

    Returns
    -------
    None.

    """
    print('Start saving results to excel')
    output_file_path = os.path.join(output_dir, 'output.xlsx')
    excel_format_res = dict()
    for name_img, tablet_plaques in tablets_results.items():
        result = pd.DataFrame(columns = range(1,7), index = ['A', 'B', 'C', 'D'])
        for cell, val in tablet_plaques.items():
            i = int(cell[1])
            c = cell[0]
            result.loc[c, i] = val
        excel_format_res[name_img] = result
    with pd.ExcelWriter(output_file_path, mode = 'w') as writer:
        for key, value in excel_format_res.items():
            value.to_excel(writer, sheet_name = key, columns = None, index_label = None)
    
def save_to_separate_excel(tablets_results: dict, output_dir: str):
    """
    Save counted plaques in tablet (on the image) into separate excel files: each file for each image.

    Parameters
    ----------
    tablet_results : dict
        dictionary where the key is the name of the image file, and the value is a dictionary where the keys are the cells of the tablet, the values are the number of plaques
    output_dir : str
        directory for saving the excel file

    Returns
    -------
    None.

    """
    print('Start saving results to excel')
    excel_format_res = dict()
    for name_img, tablet_plaques in tablets_results.items():
        result = pd.DataFrame(columns = range(1,7), index = ['A', 'B', 'C', 'D'])
        for cell, val in tablet_plaques.items():
            i = int(cell[1])
            c = cell[0]
            result.loc[c, i] = val
        excel_format_res[name_img] = result
        for name_img, value in excel_format_res.items():
            name_file = name_img + ".xlsx"
            output_file_path = os.path.join(output_dir, name_file)
            value.to_excel(output_file_path, sheet_name = name_img, columns = None, index_label = None)  

def ret_proc(val): 
    return val

def circ_detect_(dir_path, out_dir, mode, excel_mode):
    print('Start...')
    imgs_names = choose_image_file_names(dir_path)
    dict_img_cells_count = dict()
    count_img = len(imgs_names)
    step = 1
    for img_name in imgs_names:
        path_img = os.path.join(dir_path, img_name)
        initial_img = cv2.imread(path_img, -1)
        clustered_img = cluster_img(initial_img)
        masked_im, mask = mask_img(clustered_img, initial_img)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        img_res_dir = img_name.split('.')[0] + '_res/'
        path_out = os.path.join(out_dir, img_res_dir)
        os.mkdir(path_out)
        for r_max in range(1, 1000, 1):
            circles_params = detect_circles(mask, 30, r_max)
            if circles_params is not None:
                if circles_params.shape[1] == 24:
                    print(circles_params)
                    radii = circles_params[:, :, 2]
                    max_r = np.max(radii)
                    min_r = np.min(radii)
                    if (max_r - min_r) < 3:
                        if mode == 'auto':
                            out_name = os.path.join(path_out, ('maskimg_' + img_name.split(".")[0] + '.jpg'))
                            resized_img = draw_circles(mask, circles_params)
                            cv2.imwrite(out_name, resized_img)
                            act_step = int(step/count_img*100)
                            step = step + 1
                            print(act_step)
                            print(step)
                            ret_proc(act_step)
                            break
                        if mode == 'handle':
                            visualise_circle(mask, circles_params)
                            print('Enter y if detect is correct:')
                            norm = input()
                            if norm == 'y':
                                break    
        Lunki=dictMaker(circles_params)
        diction = calc_obj(Lunki, masked_im, path_out)
        dict_img_cells_count[img_name] = diction
    print(dict_img_cells_count)
    if not bool(dict_img_cells_count):
        raise ValueError('ValueError 328: Could not find any plaques')
    print(bool(dict_img_cells_count))
    if excel_mode == "uni" and bool(dict_img_cells_count):
        print(dict_img_cells_count)
        save_to_one_excel(dict_img_cells_count, out_dir)
    elif excel_mode == "many" and bool(dict_img_cells_count):
        save_to_separate_excel(dict_img_cells_count, out_dir)
    else:
        print("Error")
    print('Finish!')


class ProcThread(QThread):
    progress = pyqtSignal(int)

    def __init__(self):
        super(ProcThread, self).__init__()
        self.dir_path = None
        self.out_dir = None
        self.mode = None
        self.excel_mode = None

    def setParams(self, dir_path, out_dir, mode, excel_mode):
        self.dir_path = dir_path
        self.out_dir = out_dir
        self.mode = mode
        self.excel_mode = excel_mode

    def circ_detect(self, dir_path, out_dir, mode, excel_mode):
        print('Start...')
        imgs_names = choose_image_file_names(dir_path)
        dict_img_cells_count = dict()
        count_img = len(imgs_names)
        step = 1
        for img_name in imgs_names:
            path_img = os.path.join(dir_path, img_name)
            initial_img = cv2.imread(path_img, -1)
            clustered_img = cluster_img(initial_img)
            masked_im, mask = mask_img(clustered_img, initial_img)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            img_res_dir = img_name.split('.')[0] + '_res/'
            path_out = os.path.join(out_dir, img_res_dir)
            os.mkdir(path_out)
            for r_max in range(1, 1000, 1):
                circles_params = detect_circles(mask, 30, r_max)
                if circles_params is not None:
                    if circles_params.shape[1] == 24:
                        print(circles_params)
                        radii = circles_params[:, :, 2]
                        max_r = np.max(radii)
                        min_r = np.min(radii)
                        if (max_r - min_r) < 3:
                            if mode == 'auto':
                                out_name = os.path.join(path_out, ('maskimg_' + img_name.split(".")[0] + '.jpg'))
                                resized_img = draw_circles(mask, circles_params)
                                cv2.imwrite(out_name, resized_img)
                                act_step = int(step/count_img*100)
                                step = step + 1
                                print(act_step)
                                print(step)
                                self.progress.emit(act_step)
                                break
                            if mode == 'handle':
                                visualise_circle(mask, circles_params)
                                print('Enter y if detect is correct:')
                                norm = input()
                                if norm == 'y':
                                    act_step = int(step/count_img*100)
                                    step = step + 1
                                    self.progress.emit(act_step)
                                    break    
            Lunki=dictMaker(circles_params)
            diction = calc_obj(Lunki, masked_im, path_out)
            dict_img_cells_count[img_name] = diction
        print(dict_img_cells_count)
        if not bool(dict_img_cells_count):
            raise ValueError('ValueError 328: Could not find any plaques')
        print(bool(dict_img_cells_count))
        if excel_mode == "uni" and bool(dict_img_cells_count):
            print(dict_img_cells_count)
            save_to_one_excel(dict_img_cells_count, out_dir)
        elif excel_mode == "many" and bool(dict_img_cells_count):
            save_to_separate_excel(dict_img_cells_count, out_dir)
        else:
            print("Error")
        print('Finish!')

    def run(self):
        print(self.dir_path)
        self.circ_detect(self.dir_path, self.out_dir, self.mode, self.excel_mode)



class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('mywindow_1.ui', self)
        self.pushButton.clicked.connect(self.enterInDir)
        self.pushButton_2.clicked.connect(self.enterOutDir)
        self.pushButton_3.clicked.connect(self.startProc)
        self.mode = 'auto'
        self.one_excel = 'many'
        self.indir = None
        self.outdir = None
        self.pt = ProcThread()
        self.pt.progress.connect(self.updateProgressBar)
        self.show()

    def enterInDir(self):
        dw = QFileDialog()
        self.indir = dw.getExistingDirectory(self, 'Select the directory with the images')

    def enterOutDir(self):
        dw = QFileDialog()
        self.outdir = dw.getExistingDirectory(self, 'Select the directory to record the results in')

    def updateProgressBar(self, value):
        print('Work ' + str(value))
        self.progressBar.setValue(value)
    
    def startProc(self):
        self.progressBar.setValue(0)
        if self.indir != None:
            if self.outdir != None:
                mode_st = self.checkBox.isChecked()
                mode = self.mode
                one_excel = self.one_excel
                if mode_st == True:
                    mode = 'handle'
                mode_ex = self.checkBox_2.isChecked()
                if mode_ex == True:
                    one_excel = 'uni'
                self.pt.setParams(self.indir, self.outdir, mode, one_excel)
                self.pt.start()
                
            else:
                box = QMessageBox()
                box.setWindowTitle('Attention!')
                box.setText('The directory for saving the results is not selected')
                box.exec()
        else:
            box = QMessageBox()
            box.setWindowTitle('Attention!')
            box.setText('The image directory has not been selected for processing')
            box.exec()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())

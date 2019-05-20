import numpy as np 
import cv2
import math

fway = 'egtest03/frame0'
pway = 'egtest03_masks/mask0'

def makeRoad(state,flag,type):
	back = str(flag)
	while len(back) < 4:
		back = '0' + back
	if type == 1:
		result = state + back + '.jpg'
	else:
		result = state + back + '.txt'
	return result

#get ix,iy,fx,fy
def getPoint(road):
	file = open(road)
	s = file.readline()
	s = file.readline()
	i = 0
	list = []
	num = ''
	while i < len(s):
		if s[i] == ' ':
			list.append(int(num))
			num = ''
		else:
			num += s[i]
		i+=1
	return list[2],list[0],list[3],list[1]

def calCenter(points):
	xcenter = points[:,0].sum()/points.shape[0]
	ycenter = points[:,1].sum()/points.shape[0]
	return xcenter,ycenter

def dropBad(find_old,find_new,xcenter,ycenter,T):
	while(1):
		waitForDrop = []
		for i,point in enumerate(find_new):
			xpoint,ypoint = point.ravel()
			distance = pow(xpoint-xcenter,2) + pow(ypoint-ycenter,2)
			if distance > pow(T,2):
				waitForDrop.append(i)
		#没有需要淘汰的特征点时，循环结束
		if len(waitForDrop) == 0:
			break
		#去除不符合条件的特征点
		find_old = np.delete(find_old,waitForDrop,0)
		find_new = np.delete(find_new,waitForDrop,0)
	return find_old,find_new

def calMSE(predict,target,M,N):
	tmp = (target - predict) 
	amount = (tmp*tmp).sum()
	mse = math.sqrt(amount/M*N)
	return mse

def targetTrace(feature_params,lk_params,base):
	flag = 0
	frame = cv2.imread(makeRoad(fway,flag+base,1),-1)
	fitness = 0
	pixel_predict = 0
	radius = 0
	while flag<=10:
		old_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		if flag % 10 == 0:
			mask0 = np.zeros_like(old_gray)
			ix,iy,fx,fy = getPoint(makeRoad(pway,flag+base,2))
			mask0[iy:fy,ix:fx] = 255

			radius = (fx-ix)/2
			xLength = (fx-ix)/2
			yLength = (fy-iy)/2
			if flag != 0: 
				#找质心，划框
				target = old_gray[iy:fy,ix:fx]
				mask2 = np.zeros_like(frame)
				mask2 = cv2.rectangle(mask2,(ix,iy),(fx,fy),(0,255,0),3)
				pic = cv2.add(frame,mask2)
				mask3 = np.zeros_like(frame)
				centerX = find_new[:,0].sum()/find_new.shape[0]
				centerY = find_new[:,1].sum()/find_new.shape[0]
				ix,iy = int(centerX-xLength),int(centerY-yLength)
				fx,fy = int(centerX+xLength),int(centerY+yLength)
				predict = old_gray[iy:fy,ix:fx]
				mask3 = cv2.rectangle(mask3,(ix,iy),(fx,fy),(0,255,0),3)
				pic_ = cv2.add(frame,mask3)
				#cv2.imshow('target',pic)
				#cv2.imshow('predict',pic_)
				#cv2.waitKey(30)
				mse = calMSE(predict,target,fx-ix,fy-iy)
				#print(mse)
			else:
				old_corner = cv2.goodFeaturesToTrack(old_gray,mask=mask0,**feature_params)
		else:
			old_corner = find_new.reshape(find_new.shape[0],1,find_new.shape[1])
		
		#get next frame
		flag += 1
		frame = cv2.imread(makeRoad(fway,flag+base,1),-1)
		new_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		new_corner,trace_st,err = cv2.calcOpticalFlowPyrLK(old_gray,new_gray,old_corner,None,**lk_params)
		
		try:
			#get matched points
			find_new = new_corner[trace_st==1]
			find_old = old_corner[trace_st==1]

			#calculate center and drop bad points
			xcenter_old,ycenter_old = calCenter(find_old)
			find_old,find_new = dropBad(find_old,find_new,xcenter_old,ycenter_old,radius)
			
		except:
			return 1e9,0
	
	return mse,find_new.shape[0]

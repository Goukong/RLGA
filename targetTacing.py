import numpy as np 
import cv2

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

def calMSE():
	return

def targetTrace(feature_params,lk_params):
	flag = 0
	frame = cv2.imread(makeRoad(fway,flag,1),-1)
	fitness = 0
	pixel_predict = 0
	radius = 0
	while flag<=100:
		old_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		if flag % 10 == 0:
			mask0 = np.zeros_like(old_gray)
			ix,iy,fx,fy = getPoint(makeRoad(pway,flag,2))
			mask0[iy:fy,ix:fx] = 255
			
			radius = (fx-ix)/2
			if flag != 0: 
				#print(find_new.shape)
				centerX = int((fx+ix)/2)
				centerY = int((fy+iy)/2)
				pixel_target = old_gray[centerY,centerX]
				if pixel_target == pixel_predict:
					fitness += 1
			else:
				old_corner = cv2.goodFeaturesToTrack(old_gray,mask=mask0,**feature_params)
		else:
			old_corner = find_new.reshape(find_new.shape[0],1,find_new.shape[1])
		
		#get next frame
		flag += 1
		frame = cv2.imread(makeRoad(fway,flag,1),-1)
		new_gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		new_corner,trace_st,err = cv2.calcOpticalFlowPyrLK(old_gray,new_gray,old_corner,None,**lk_params)
		
		try:
			#get matched points
			find_new = new_corner[trace_st==1]
			find_old = old_corner[trace_st==1]

			#calculate center and drop bad points
			xcenter_old,ycenter_old = calCenter(find_old)
			find_old,find_new = dropBad(find_old,find_new,xcenter_old,ycenter_old,radius)
			#after dropping ,get the new center
			xcenter_new,ycenter_new = calCenter(find_new)
			pixel_predict = new_gray[int(ycenter_new),int(xcenter_new)]
		except:
			return 0
	
	return find_new.shape[0]

import cv2 as cv
import numpy as np
import pyopencl as cl
import  imageForms as iF
import math
import car_detection_video as car_det
import matplotlib.pyplot as plt

# (1) setup OpenCL
try:
    plaforms= cl.get_platforms()
    global plaform
    plaform= plaforms[0]

    devices = plaform.get_devices()
    global device
    device= devices[0]

    global ctx
    ctx= cl.Context(devices) # or dev_type=cl.device_type.ALL)

    global commQ
    commQ= cl.CommandQueue(ctx,device)

    file = open("hough.cl","r")
    global hough
    hough= cl.Program(ctx,file.read())
    hough.build()



    pathname = "C:\\Users\\ragma\\OneDrive\\Desktop\\TAPDI\\images\\video\\"

    filename = "video1.MTS"
    # get video
    cap = cv.VideoCapture(pathname + filename)

    if not cap.isOpened():
        print("Video File Not Found")
        exit(-1)

    # get video frame by frame
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # resizing for faster detection
        frame = cv.resize(frame, (840, 440))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        imgOut = np.copy(frame)
        imgOut_1 = np.copy(frame)

        # (2) get shape of input image, allocate memory for output to which result can be copied to
        height = frame.shape[0]
        width = frame.shape[1]
        padding = frame.strides[0] - frame.shape[1] * frame.strides[1]
        arrayIn = np.zeros((round(math.sqrt(width * width + height * height)), 180), dtype=np.int32)
        arrayIn_1 = np.zeros((round(math.sqrt(width * width + height * height)), 180), dtype=np.int32)
        #math.sqrt((width*width) + (height*height))

        # (2) create image buffers which hold images for OpenCL
        kernelName = hough.line_seg_image2D
        kernelName_1 = hough.sobel_image2D
        kernelName_2 = hough.grey_image2D
        kernelName_3 = hough.binary_threshold_image2D
        imgFormat = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        imgInBuffer = cl.Image(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY,
                           format=imgFormat, shape=(width, height), pitches=(frame.strides[0], frame.strides[1]),
                           hostbuf=frame.data)
        imgOutBuffer = cl.Image(ctx, flags=cl.mem_flags.ALLOC_HOST_PTR | cl.mem_flags.READ_WRITE,
                            format=imgFormat, shape=(width, height))
        imgOutBuffer_1 = cl.Image(ctx, flags=cl.mem_flags.ALLOC_HOST_PTR | cl.mem_flags.READ_WRITE,
                                format=imgFormat, shape=(width, height))
        memBuffer = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf=arrayIn)
        memBuffer_1 = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf=arrayIn_1)

        kernelName_2.set_arg(0, imgInBuffer)
        kernelName_2.set_arg(1, imgOutBuffer)
        kernelName_2.set_arg(2, np.int32(width))  # Width
        kernelName_2.set_arg(3, np.int32(height))  # Height

        kernelName_3.set_arg(0, imgOutBuffer)
        kernelName_3.set_arg(1, imgOutBuffer_1)
        kernelName_3.set_arg(2, np.int32(width))  # Width
        kernelName_3.set_arg(3, np.int32(height))  # Height
        kernelName_3.set_arg(4, np.int32(180))

        kernelName.set_arg(0, imgOutBuffer_1)
        kernelName.set_arg(1, np.int32(frame.shape[1]))  # Width
        kernelName.set_arg(2, np.int32(frame.shape[0]))  # Height
        kernelName.set_arg(3, np.int32(padding))         # Padding
        kernelName.set_arg(4, np.int32(255))               # Threshold
        kernelName.set_arg(5, memBuffer)
        kernelName.set_arg(6, memBuffer_1)


        workGroupSize = (math.ceil(np.int32(frame.shape[1]) / 32) * 32, math.ceil(np.int32(frame.shape[0]) / 8) * 8)
        workItemSize = (32, 8)  # 1024

        kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName_2, global_work_size=workGroupSize,
                                             local_work_size=workItemSize)
        kernelEvent.wait()
        
        kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName_3, global_work_size=workGroupSize,
                                                 local_work_size=workItemSize)
        kernelEvent.wait()

        # (4) copy image to device, execute kernel, copy data back
        cl.enqueue_copy(commQ, imgOut, imgOutBuffer_1, origin=(0, 0), region=(width, height))
        #print(workGroupSize)


        #HOUgh
        kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName, global_work_size=workGroupSize,
                                         local_work_size=workItemSize)
        kernelEvent.wait()

        # get buffer for rhos &thetas
        cl.enqueue_copy(commQ, arrayIn, memBuffer)
        cl.enqueue_copy(commQ, arrayIn_1, memBuffer_1)
        #print(arrayIn)
        # get indez of most voted on in array
        most_votes = np.max(arrayIn)

        most_votes_1 = np.max(arrayIn_1)
        # most_votes_index = arrayIn.index(most_votes)
        most_votes_index = np.argmax(arrayIn == most_votes)
        most_votes_index_1 = np.argmax(arrayIn_1 == most_votes_1)


        # get rha and theta
        # place = rho * 180 + theta
        rho = (most_votes_index // 180 )- 948/2;
        theta = most_votes_index % 180;
        a = math.cos(math.radians(theta))
        b = math.sin(math.radians(theta))
        x0 = a * rho
        y0 = b * rho

        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        m = (float) ((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))
        b = (pt2[1]-(m * pt2[0]))
        print(m, b)
        cv.line(frame, pt1, pt2, (255, 0, 0), 1, cv.LINE_AA)


        # get rha and theta
        # place = rho * 180 + theta
        rho_1 = most_votes_index_1 // 180 -  948/2;
        theta_1 = most_votes_index_1 % 180;
        a_1 = math.cos(math.radians(theta_1))
        b_1 = math.sin(math.radians(theta_1))
        x0_1 = a_1 * rho_1
        y0_1 = b_1 * rho_1

        pt1_1 = (int(x0_1 + 1000 * (-b_1)), int(y0_1 + 1000 * (a_1)))
        pt2_1 = (int(x0_1 - 1000 * (-b_1)), int(y0_1 - 1000 * (a_1)))
        if(pt2_1[0] - pt1_1[0]) != 0 :
            m_1 = (float)((pt2_1[1] - pt1_1[1]) / (pt2_1[0] - pt1_1[0]))
            b_1 = (pt2_1[1] - (m_1 * pt2_1[0]))
            print(m_1, b_1)
        else:
            m_1=0
            b_1=0
            print("zero")

        cv.line(frame, pt1_1, pt2_1, (255, 0, 0), 1, cv.LINE_AA)

        imgInBuffer.release()
        imgOutBuffer.release()

        car_det.detectAndDisplay(frame, m, b, m_1, b_1)
        # Display the resulting frame
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('p'):
            plt.imshow(imgOut)
            plt.show()
            

        if cv.waitKey(1) & 0xFF == ord('q'):
            break



    # When everything done, release the capture
    imgInBuffer.release()
    imgOutBuffer.release()
    memBuffer.release()
    memBuffer_1.release()
    cap.release()

except Exception as e:
    print(e)
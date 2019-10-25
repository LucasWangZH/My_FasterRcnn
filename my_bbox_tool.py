import numpy as np

def bbox_iou(src_bbox,dst_bbox):
    """
    calc iou between bboxes
    :param src_bbox: (N,4) y1x1,y2x2
    :param dst_bbox: (K,4)
    :return: (N,K)
    """
    #iou = np.zeros(src_bbox.shape[0],dst_bbox.shape[0]).astype(np.float32)
    # a_lt = src_bbox[:,:2]
    # a_br = src_bbox[:,2:]
    # b_lt = dst_bbox[:,:2]
    # b_br = dst_bbox[:,2:]
    # area_a = np.prod((a_br - a_lt),axis = 1)
    # area_a = np.repeat(area_a,axis = 0).reshape(iou.shape)
    #
    # area_b = np.prod((b_br - b_lt),axis = 1).reshape(iou.shape[1],1)#k,1
    # area_b = np.repeat(area_b,axis = 0).reshape(iou.shape)
    #
    # iou = area_a + area_b#N,K
    #
    # for i in range(src_bbox):
    #     for j in range(dst_bbox):
    #         bboxa = src_bbox[i,:]
    #         bboxb = dst_bbox[j,:]
    #         x1 = np.maximum(bboxa[1],bboxb[1])
    #         y1 = np.maximum(bboxa[0],bboxb[0])
    #         x2 = np.minimum(bboxa[3],bboxb[3])
    #         y2 = np.maximum(bboxa[2],bboxb[2])
    #
    #         area_intersect = (y2 - y1) * (x2 - x1)
    #         if area_intersect > 0:
    #             iou[i][j] = area_intersect / iou[i][j]
    #         else:
    #             iou[i][j] = 0
    # return iou
    if len(dst_bbox.shape) == 3:
        dst_bbox = dst_bbox[0,:,:]
    elif len(dst_bbox.shape) == 2:
        dst_bbox =dst_bbox
    elif len(dst_bbox.shape) > 3:
        raise IndexError

    if src_bbox.shape[1] != 4 or dst_bbox.shape[1] != 4:
        raise IndexError

    #srcbbox和dstbbox比较，运用广播机制，出来N，K，2
    lt = np.maximum(src_bbox[:,None,:2],dst_bbox[None,:,:2])
    br = np.minimum(src_bbox[:,None,2:],dst_bbox[None,:,2:])

    # if lt !< br, then 0
    #如果lt not < br,就是0
    area_i = np.prod(br - lt,axis = 2) * (lt < br).all(axis = 2)
    area_a = np.prod(src_bbox[:,2:] - src_bbox[:,:2],axis = 1)
    area_b = np.prod(dst_bbox[:,2:] - dst_bbox[:,:2],axis = 1)
    #广播除法,自动补充维度
    iou = area_i / ((area_a[:,None] + area_b) - area_i)
    return iou

def bbox2loc(src_bbox,dst_bbox):
    """
    encode bbox to loc(offsets and scales)
    :param src_bbox: array(R,4)
    :param dst_bbox: array(R,4)
    :return: array(R,4) loc, loc contains offsets and scales.
    The second axis contains four values :math:`t_y, t_x, t_h, t_w`.
    :Formula: dy = (dst_bbox.ctry - src_bbox.ctry)/ src_bbox.height
              dx = (dst_bbox.ctrx - src_bbox.ctrx)/ src_bbox.widht
              dh = log(dst.height / src.height)
              dw = log(dst.width / src.width）
    """

    src_bbox_height = src_bbox[:,2] - src_bbox[:,0]
    src_bbox_width = src_bbox[:,3] - src_bbox[:,1]
    src_bbox_ctry = src_bbox[:,0] + 0.5 * src_bbox_height
    src_bbox_ctrx = src_bbox[:,1] + 0.5 * src_bbox_width

    dst_bbox_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    dst_bbox_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    dst_bbox_ctry = dst_bbox[:, 0] + 0.5 * dst_bbox_height
    dst_bbox_ctrx = dst_bbox[:, 1] + 0.5 * dst_bbox_width


    #用eps处理掉0和负数
    eps = np.finfo(src_bbox_height.dtype).eps
    src_bbox_height = np.maximum(src_bbox_height,eps)
    src_bbox_width = np.maximum(src_bbox_width,eps)


    dy = (dst_bbox_ctry - src_bbox_ctry) / src_bbox_height
    dx = (dst_bbox_ctrx - src_bbox_ctrx) / src_bbox_width
    dh = np.log(dst_bbox_height / src_bbox_height)
    dw = np.log(dst_bbox_width / src_bbox_width)

    loc = np.concatenate((dy[:,None],dx[:,None],dh[:,None],dw[:,None]),axis=1)
    return loc

def loc2bbox(src_bbox,loc):
    """
    Decode bbox from location，loc is the offset. Given one box and one offset,we can get a target box(the coordiantes in 2d pic)
    loc是偏移量，就是给一个框，一个偏移量，出个目标框(就是给2d图中的坐标了)
    :param src_bbox: array(R,4)[lty,ltx,bry,brx],R is the number of boxes
    :param loc: array(R,4)[dy,dx,dh,dw](也就是t_y,t_x,t_h,t_w)
    :return:array(R,4) dst_box [lty,ltx,bry,brx]
    :Formula: center_y = dy*src_bbox.height + src_ctr_y
             center_x = dx*src_bbox.weidth + src_ctr_x
             h = exp(dh) * src_bbox.height
             w = exp(dw) * src_bbox.width
             dst_bbox.lty = center_y - 0.5 * h
             dst_bbox.ltx = center_x - 0.5 * w
             dst_bbox.bry = center_y + 0.5 * h
             dst_bbox.brx = center_x + 0.5 * w

    """
    dst_bbox = np.zeros((src_bbox.shape),dtype= np.float32)

    src_bbox_height = src_bbox[:,2] - src_bbox[:,0]
    src_bbox_width = src_bbox[:,3] - src_bbox[:,1]
    src_bbox_ctry = src_bbox[:, 0] + 0.5 * src_bbox_height
    src_bbox_ctrx = src_bbox[:, 1] + 0.5 * src_bbox_width

    dst_cty = loc[:,0] * src_bbox_height + src_bbox_ctry
    dst_ctx = loc[:,1] * src_bbox_width + src_bbox_ctrx
    h = np.exp(loc[:,2]) * src_bbox_height
    w = np.exp(loc[:,3]) * src_bbox_width

    dst_bbox[:,0] = dst_cty - 0.5 * h
    dst_bbox[:,1] = dst_ctx - 0.5 * w
    dst_bbox[:,2] = dst_cty + 0.5 * h
    dst_bbox[:,3] = dst_ctx + 0.5 * w

    return dst_bbox

def get_inside_index(anchor, H, W):
    # retrive the indexed of all the boxes that has all 4 coordinates inside the imgsize
    #获取所有 4个坐标都在imgsize内部的bbox的index
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


def unmap(data,count,index,fill = 0):
    #unmap a subset of item(data) back to the original set of items(of size count)

    if len(data.shape) == 1:
        ret = np.empty((count,),dtype= data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret



def base_anchor_generator(base_size = 16,ratios = [0.5,1,2], scales = [8,16,32]):
    """
    generate 9 base anchor, at (0,0) position, then shift it to generate that for the whole pic
    生成9个base anchor，在（0，0）处，后面做漂移生成全图的
    :param base_size:
    :param ratios:
    :param scales:
    :return:
    """
    ctrx = base_size / 2.
    ctry = base_size / 2.

    anchor_base = np.zeros(((len(ratios) * len(scales)),4),dtype = np.float32)
    len_ratios = len(ratios)
    for i in range(len(ratios)):
        for j in range(len(scales)):
            H = base_size * scales[i] * np.sqrt(ratios[j])
            W = base_size * scales[i] * np.sqrt(ratios[len_ratios -1 - j])

            anchor_base[i * len_ratios + j][0] = ctry - H / 2.
            anchor_base[i * len_ratios + j][1] = ctrx - W / 2.
            anchor_base[i * len_ratios + j][2] = ctry + H / 2.
            anchor_base[i * len_ratios + j][3] = ctrx + W / 2.

    return anchor_base

def enumerate_shift_anchor(anchor_base,feat_stride,height,width):
    shift_y = np.arange(0,height * feat_stride,feat_stride)
    shift_x = np.arange(0,width * feat_stride, feat_stride)
    #shift_x is (w,1) shift_y is (h,1)
    #after meshgrid,shift_x and shift_y are (w,h)
    shift_x,shift_y = np.meshgrid(shift_x,shift_y)

    #shift(w*h,4)
    shift = np.stack((shift_y.ravel(),shift_x.ravel(),shift_y.ravel(),shift_x.ravel()),axis = 1 )

    A = anchor_base.shape[0]
    K = shift.shape[0]
    #reshape anchor_base means that we add an axis, it turn to one of (A*4) 
    #anchor_base的reshape就相当于加了个轴，变成 1个A*4
    #shiftreshape is same as reshape of anchorbase
    #shiftreshape同anchorbase的reshape
    #transpose is actually change the 0th axis and 1th axis, so it's turned from one (K*4) to k (1*4), for the sake of broadcasting laterS
    #shift reshape后的transpose等同于0轴和1轴互换，就从1个k*4变成了k个1*4，方便后面的广播加法计算
    anchor = anchor_base.reshape((1,A,4)) + shift.reshape((1,K,4)).transpose((1,0,2))
    anchor = anchor.reshape((K*A,4)).astype(np.float32)#reshape成为所有anchor的矩阵形式
    return anchor
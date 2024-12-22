# THRESHOLD
CFG_THRESHOLD = 0.3

# YOLO
CFG_PATH_YOLO_MODEL = './best_7.pt'

# FASTER RCNN
CFG_PATH_FASTERRCNN_MODEL = './faster_rcnn_best.pth'

# DEVICE
CFG_DEVICE = 'cpu' # ['cpu' , 'cuda'] # FASTER RCNN

# MODEL
CFG_MODEL = None  # Default value

# CFG_MODEL = 'fasterrcnn' # ['yolo' , 'fasterrcnn']

# HTTP SERVER
CFG_HTTP_TYPE = 'local' 

# CUSTOM
CFG_ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
CFG_PATH_UPLOAD = './static/uploads'
CFG_PATH_RESULT = './static/results'
CFG_MAX_CONTENT_LENGTH = 5 * 1024 * 1024 # 5MB

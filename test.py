import torch
from ultralytics import YOLO
from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Engine, Events

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    filename='logs/test.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Инициализация модели YOLO
model = YOLO('runs/detect/train/weights/best.pt')

# Директория с изображениями
input_dir = Path('dataset/test/images')

# Настройка TensorBoard
writer = SummaryWriter(log_dir='logs/tensorboard')

# Функция обработки изображений
def process_function(engine, batch):
    img_path = batch
    results = model(img_path, save=True)
    num_detections = len(results[0])  # Число объектов на изображении
    logging.info(f'Image: {img_path.name} | Detections: {num_detections}')
    
    # Логирование в TensorBoard
    writer.add_scalar('Detections per Image', num_detections, engine.state.iteration)
    return num_detections

# Создание объекта Engine
engine = Engine(process_function)

# Событие завершения обработки
@engine.on(Events.COMPLETED)
def on_completed(engine):
    logging.info("Процесс обнаружения завершен.")
    writer.close()

# Загрузка данных
data = list(input_dir.iterdir())

# Запуск обработки
engine.run(data)

# metric P R F
import torch


class PRFMetric:
    def __init__(self, num_classes, label_type) -> None:
        self.correct = 0
        self.all = 0
        self.num_classes = num_classes
        self.label_type = label_type
        self.best_preciscion = 0
        self.best_recall = 0
        self.best_f1 = 0
        self.best_acc = 0
        if self.label_type == "authScore":
          self.num_classes = 2
        self.tp = torch.ones(self.num_classes)
        self.fp = torch.ones(self.num_classes)
        self.fn = torch.ones(self.num_classes)
        '''
        if self.label_type == "authScore":
          self.tp = torch.ones(2)
          self.fp = torch.ones(2)
          self.fn = torch.ones(2)
        '''

    def add_data_piece(self, logits, labels):
        logits = logits.detach().to(torch.device("cpu"))
        labels = labels.detach().to(torch.device("cpu"))
        predictions = torch.argmax(logits, dim=1)
        
        self.correct += torch.sum(predictions == labels).item()
        self.all += labels.shape[0]
        # num_classes = len(torch.unique(labels))

        if self.label_type == "authScore":
          ones = torch.ones_like(predictions)
          predictions = torch.where(predictions > 1, ones, predictions)
        
        for c in range(self.num_classes):
          self.tp[c] += ((predictions == c) & (labels == c)).sum()
          self.fp[c] += ((predictions == c) & (labels != c)).sum()
          self.fn[c] += ((predictions != c) & (labels == c)).sum()

        '''
        if self.label_type != "authScore":
          for c in range(num_classes):
              self.tp[c] += ((predictions == c) & (labels == c)).sum()
              self.fp[c] += ((predictions == c) & (labels != c)).sum()
              self.fn[c] += ((predictions != c) & (labels == c)).sum()
        elif self.label_type == "authScore":
          ones = torch.ones_like(predictions)
          predictions = torch.where(predictions > 1, ones, predictions)
          
          for c in range(2):
              self.tp[c] += ((predictions == c) & (labels == c)).sum()
              self.fp[c] += ((predictions == c) & (labels != c)).sum()
              self.fn[c] += ((predictions != c) & (labels == c)).sum()
        '''



    def get_metric(self):
        precision = self.tp / (self.tp + self.fp)
        precision = torch.mean(precision)
        recall = self.tp / (self.tp + self.fn)
        recall = torch.mean(recall)
        f1 = 2 * precision * recall / (precision + recall)
        acc = self.correct / self.all

        self.best_preciscion = max(self.best_preciscion, precision)
        if f1 > self.best_f1:
            self.best_f1 = f1
            # self.best_preciscion = precision
            self.best_recall = recall
            self.best_acc = acc
        return {
            "presicion": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "acc": acc,
        }

    def get_best_metric(self):
        return {
            "presicion": self.best_preciscion,
            "recall": self.best_recall,
            "f1": self.best_f1,
            "acc": self.best_acc,
        }

    def get_best_metric_2_logger_format(self):
        logger_format = "P:{:.4f} R:{:.4f} F1:{:.4f} ACC:{:.4f}"
        return logger_format.format(
            self.best_preciscion, self.best_recall, self.best_f1, self.best_acc
        )

    def reset_epoch(self):
        self.correct = 0
        self.tp = torch.ones(self.num_classes)
        self.fp = torch.ones(self.num_classes)
        self.fn = torch.ones(self.num_classes)
        self.all = 0

    def reset_all(self):
        self.correct = 0
        self.tp = torch.ones(self.num_classes)
        self.fp = torch.ones(self.num_classes)
        self.fn = torch.ones(self.num_classes)
        self.all = 0

        self.best_preciscion = 0
        self.best_recall = 0
        self.best_f1 = 0
        self.best_acc = 0

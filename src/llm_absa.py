#implementation2



classifier = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")
sentence = "The food was exceptional, although the service was a bit slow."


# Düzeltmeler ve doğrulama

Kaynak commit: `5cec2cfb85def65d7d61b35d533c8b6ac20940ec`.

Bu sürümde aşağıdaki çalışma zamanı ve eğitim hataları düzeltildi:

- Test paketinin eksik `__init__.py` dosyası ve encoder `forward()` metodu.
- Eksik `torch.nn.functional` importu, query cache ayrıştırması ve eğitim
  sırasında eski hesaplama grafiğinin cache'den kullanılmasının önlenmesi.
- Gerçek EmbeddingGemma 2'nin 768 boyutlu projeksiyon çıktısı ile 512 boyutlu
  transformer token state'lerinin ayrılması; çoklu katman seçiminde fusion boyutu.
- Native embedding ortalamasında padding maskesi, cihaz ve dtype uyumu.
- Dot fusion'da yanlış boyut ve her çağrıda parametrelerin yeniden oluşturulması.
- Token interaction boyutları, padding maskeleri ve token bütçeleri; MaxSim
  için ortak query/document projeksiyonu.
- Confidence boyutları, inference sonuçlarına confidence eklenmesi ve iki adaylı
  listelerde sabit confidence üreten normalizasyonun kaldırılması.
- Chunking tokenizer desteği, prompt için token bütçesi ayırma ve gerçek içerik
  kaybının `truncated_documents` alanında raporlanması.
- Aynı document ID'sini kullanan adaylarda doğru belgenin döndürülmesi.
- Pairwise loss'ta aday sırasından bağımsız graded relevance karşılaştırması;
  listwise/distillation hedef dağılımlarında padding'in dışlanması.
- Ragged eğitim batch'lerinde yalnızca gerçek belgelerin encode edilmesi ve
  her query'nin kendi adaylarıyla eşleştirilmesi.
- Varsayılan backbone dondurma, LoRA named-parameter kontrolü ve gradient
  accumulation sonunda eksik kalan optimizer adımının uygulanması.
- Evaluation confidence çağrısı, graded relevance korunması ve rapor formatı.
- Yanlış iki aritmetik test beklentisi ve öğretmen skorlarının testteki şekli.
- Upstream text attention head sayısının dokümanlarda 4 olarak düzeltilmesi.

## Kurulum

Python 3.12 ile doğrulandı. CPU kurulumu:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

Doğrulama ortamı: Python 3.12.14, PyTorch 2.14.1+cpu,
Transformers 5.19.0, pytest 9.1.1. GPU kullanılmadı.

```bash
pytest -q
python scripts/train_reranker.py --smoke-test
python examples/rerank_documents.py
python examples/code_rerank.py
python examples/semantic_search_rerank.py
```

140 test başarılı. Eğitim/save/reload/evaluation yolu regression testiyle
doğrulandı. Üç inference örneği gerçek `google/embeddinggemma-2` backbone'u ile
doğrulandı. CPU örneklerini daha düşük thread sayısıyla çalıştırmak için:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python examples/rerank_documents.py
```

## Model durumu ve ZIP içeriği

ZIP kaynak kodu, testleri, örnekleri, konfigürasyonları ve bu notları içerir.
Model ağırlıkları, `.git`, Python cache'leri ve smoke training çıktıları dahil
değildir. Model ilk çalıştırmada Hugging Face'den indirilir.

Reranker head'leri gerçek bir dataset üzerinde henüz eğitilmedi; hazır, eğitilmiş
reranker checkpoint'i bu ZIP'te bulunmaz. Çalışan inference ve başarılı testler
sıralama kalitesini kanıtlamaz. Üretimde kullanmak için kendi relevance verinizle
eğitim ve baseline'a karşı değerlendirme gereklidir. Confidence head'i için ayrı
supervision eklenmedi; confidence çıktıları kalibre edilmiş güven olarak
değerlendirilmemelidir.

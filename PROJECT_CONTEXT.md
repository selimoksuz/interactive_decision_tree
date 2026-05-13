# Project Context

Bu proje, Codex ile gelistirilen Streamlit tabanli interaktif decision tree uygulamasidir.

## Ana Dosyalar

- `interactive_decision_tree_app.py`: Ana Streamlit uygulamasi.
- `requirements.txt`: Python bagimliliklari.
- `run_app.ps1`: PowerShell uzerinden uygulamayi baslatir.
- `README.md`: Kurulum ve calistirma notlari.

## Calistirma

```powershell
cd C:\Users\Acer\interactive_decision_tree
.\run_app.ps1
```

Uygulama varsayilan olarak `http://localhost:8501` adresinde calisir.

## Mevcut Ozellikler

- Manuel leaf/node secimi ile interaktif agac dallandirma.
- Candidate split degiskenlerini information gain'e gore siralama.
- Numeric threshold, numeric multiway, kategorik split ve target-profile kategorik grup splitleri.
- Binary target icin target seciminin hemen altinda kullanici tarafindan secilen `Positive class`.
- Leaf/node seviyesinde prediction, impurity, default rate, event count bilgileri.
- Model performans metrikleri: binary icin AUC/Gini/accuracy/default rate, regresyon icin RMSE/MAE/R2.
- Tek tusla optimal tree kurma ve sonrasinda istedigin node'u revize etme.
- Undo last split ve reset tree.
- Zoom/fit kontrollu tree view.
- `work_id` URL parametresi ve `.tree_checkpoints/` altindaki otomatik checkpoint ile refresh sonrasi kaldigin agaci geri yukleme.
- Runnable nested tree JSON export.

## Onemli Notlar

- `Positive class` secimi default rate, event count, AUC/Gini, kategorik target-rate gruplama ve JSON export icin ortak referanstir.
- Entropy / information gain hesabi class dagiliminin safligina bakar; positive class sadece binary metriklerin ve target-rate yorumunun yonunu belirler.
- Eski gecici Codex dizinindeki kaynak dosyalar bu proje klasorune tasindi. Bundan sonraki calisma kok dizini `C:\Users\Acer\interactive_decision_tree` olmalidir.

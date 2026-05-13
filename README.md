# Interactive Decision Tree

Streamlit tabanli, manuel revize edilebilir entropy decision tree uygulamasi.

## Calistirma

PowerShell:

```powershell
cd C:\Users\Acer\interactive_decision_tree
.\.venv\Scripts\python.exe -m streamlit run .\interactive_decision_tree_app.py --server.port 8501
```

Ilk kurulum gerekiyorsa:

```powershell
cd C:\Users\Acer\interactive_decision_tree
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

## Ozellikler

- Leaf/node uzerinden manuel split secimi
- Candidate degiskenleri information gain'e gore siralama
- Numeric, kategorik, kategorik target-profile grup splitleri
- Binary target icin kullanici tarafindan secilen positive class
- Default rate, AUC, Gini ve agac toplam gain metrikleri
- Optimal tree olusturup istedigin node'dan revize etme
- Sayfa yenilense bile ayni `work_id` URL parametresiyle agaci otomatik geri yukleme
- Runnable nested tree JSON export

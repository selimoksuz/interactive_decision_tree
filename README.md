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

Notebook icinden kullanmak icin editable kurulum:

```powershell
cd C:\Users\Acer\interactive_decision_tree
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
.\.venv\Scripts\python.exe -m ipykernel install --user --name interactive-decision-tree --display-name "Python (.venv interactive_decision_tree)"
```

Python / notebook:

```python
from interactive_decision_tree import launch_tree

url = launch_tree(df, target="risk_flag", open_browser=True)
url
```

Hazir notebook ornegi:

```text
examples/notebook_dataframe_sql_demo.ipynb
```

SQL tablosu veya query sonucu ile kullanmak icin:

```python
from interactive_decision_tree import launch_tree_sql

url = launch_tree_sql(
    "sqlite:///sample.db",
    table="customers",
    target="risk_flag",
    limit=10000,
)
url
```

`launch_tree_sql` SQLAlchemy URL veya engine kabul eder. PostgreSQL, MSSQL, Oracle gibi kaynaklar icin ilgili DB driver paketini kullanici ortaminda kurmak gerekir.

## UI veri kaynaklari nasil calisir?

### Session DataFrame

`Session DataFrame`, notebook veya SQL yukleme yardimcisinin olusturdugu lokal DataFrame snapshot'ini acar. UI'dan elle dosya secmezsin; once notebookta:

```python
from interactive_decision_tree import launch_tree

url = launch_tree(df, target="risk_flag")
url
```

Bu fonksiyon `.tree_sessions/<data_id>/data.pkl` dosyasini olusturur ve `http://localhost:8501/?data_id=...&work_id=...` linkini verir. Bu link acildiginda UI otomatik `Session DataFrame` modunda baslar.

### SQL

SQL modu Streamlit UI icinden SQLAlchemy URL ile tablo veya query sonucunu DataFrame'e cevirir. Sonuc `.tree_sessions/` altina snapshot olarak yazilir; sayfa yenilenince SQL tekrar calistirilmaz.

Oracle icin ornek:

```powershell
.\.venv\Scripts\python.exe -m pip install oracledb
```

```text
oracle+oracledb://USER:PASSWORD@HOST:1521/?service_name=SERVICE_NAME
```

UI'da `Data source = SQL`, `SQL connection = Manual SQLAlchemy URL`, sonra `Table` veya `Query` secilir. Credential kaydetmek istemiyorsan manuel URL sadece o oturumda kullanilir; checkpoint'e yazilmaz.

Kayitli baglanti icin `.streamlit/secrets.toml`:

```toml
[sql_connections]
oracle_prod = "oracle+oracledb://USER:PASSWORD@HOST:1521/?service_name=SERVICE_NAME"
```

Sonra UI'da `Saved secret connection` olarak secilebilir.

## Ozellikler

- Leaf/node uzerinden manuel split secimi
- Candidate degiskenleri information gain'e gore siralama
- Numeric, kategorik, kategorik target-profile grup splitleri
- Binary target icin kullanici tarafindan secilen positive class
- Default rate, AUC, Gini ve agac toplam gain metrikleri
- Optimal tree olusturup istedigin node'dan revize etme
- Notebook RAM'indeki pandas DataFrame'i `launch_tree(df)` ile lokal Streamlit app'e tasima
- SQLAlchemy ile SQL table/query sonucunu DataFrame olarak yukleyip agacta deneme
- Sidebar uzerinden Session DataFrame, CSV / Excel Upload, SQL ve Demo kaynaklari arasinda gecis
- UI uzerinden `.csv`, `.xlsx` ve `.xls` dosyalarini yukleyebilme
- Sayfa yenilense bile ayni `work_id` URL parametresiyle agaci otomatik geri yukleme
- Runnable nested tree JSON export

## Lokal session ve secrets

Notebook ve SQL yuklemeleri `.tree_sessions/` altinda lokal pickle snapshot olarak saklanir. Uygulama URL'deki sadece guvenli `data_id` degerini okuyarak bu snapshot'i acar; raw dosya path kabul etmez.

SQL UI icin `.streamlit/secrets.toml` icine kayitli baglantilar eklenebilir:

```toml
[sql_connections]
local_sqlite = "sqlite:///sample.db"

[connections.production]
url = "postgresql+psycopg://user:password@host:5432/db"
```

SQL credential bilgileri tree checkpoint dosyalarina yazilmaz; SQL sonucunun snapshot'i saklanir.

## Final agaci notebook'a geri alma

UI'da agaci finalize ettikten sonra en alttaki `Tree export` bolumunden iki format indirebilirsin:

- `Download runnable tree JSON`
- `Download runnable tree pickle`

Pickle dosyasini ayni veya baska bir notebook'ta acmak icin:

```python
from interactive_decision_tree import load_tree_pickle

tree_payload = load_tree_pickle("interactive_entropy_tree_runnable.pkl")
tree_payload.keys()
```

JSON indirdiysen:

```python
from interactive_decision_tree import load_tree_json

tree_payload = load_tree_json("interactive_entropy_tree_runnable.json")
```

`tree_payload["tree"]` nested karar agacini, `tree_payload["nodes"]` node listesini, `tree_payload["metrics"]` model metriklerini tutar. Pickle dosyalarini sadece kendi urettiğin guvenilir dosyalardan yukle.

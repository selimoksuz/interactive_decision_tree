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

WOE workspace varsayilan olarak dahili fallback binning ile calisir. Gercek `optbinning` motorunu da kullanmak istersen:

```powershell
cd C:\Users\Acer\interactive_decision_tree
.\.venv\Scripts\python.exe -m pip install -e ".[woe]"
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
examples/notebook_separate_train_test_demo.ipynb
```

Bu notebook lokal sample DataFrame akisini ve Oracle'a demo data yazip tekrar Oracle'dan okuyarak UI'a aktarma akisini gosterir.
Ikinci notebook ayri train/test DataFrame, CSV/Excel ve lokal SQLite SQL kaynaklarini UI'da birlikte test etmek icindir.

## Is birimi icin Windows/Linux release

Business kullaniminda kullanici Python 3.10+ kurulu bir makinede zip/tar paketini acar ve launcher'i calistirir. Kullanici `pip`, `venv` veya Streamlit komutu yazmaz.

Windows:

```text
Start Interactive Tree.bat
Open Notebook.bat
```

Linux local desktop:

```bash
chmod +x start_interactive_tree.sh open_notebook.sh
./start_interactive_tree.sh --mode local --port 8501
```

Linux remote server:

```bash
./start_interactive_tree.sh --mode server --port 8501
```

Server modunda app `0.0.0.0:<port>` uzerinden dinler ve terminalde network URL'lerini yazar. Bu mod sadece guvenli kurum ici network icin dusunulmustur.

Business release offline wheelhouse bekler:

```text
wheelhouse/windows/
wheelhouse/linux/
```

Wheelhouse hangi OS ve Python minor version ile olusturulduysa ayni OS ve Python minor version ile kullanilmalidir. Ornegin Python 3.11 ile uretilen wheelhouse, Python 3.10 makinede kullanilmaz.
Kaynak kod checkout'inda wheelhouse yoksa launcher geliştirici kolayligi icin online kurulum yapabilir; business release paketinde wheelhouse zorunludur.

Release uretimi:

```powershell
.\scripts\build_release.ps1
```

```bash
chmod +x scripts/build_release.sh
./scripts/build_release.sh
```

Windows wheelhouse Windows makinede, Linux wheelhouse Linux makinede uretilir. Release paketine `.venv`, `.tree_sessions`, `.tree_checkpoints`, `.streamlit`, `oracle_config`, log/cache ve git klasorleri dahil edilmez.

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

Notebook ve UI ayni proje klasorunden calisiyorsa session klasoru otomatik eslesir. OpenShift/Jupyter gibi ortamlarda app'i elle server modunda aciyorsan launcher ayni session klasorunu set eder:

```bash
./start_interactive_tree.sh --mode server --port 8501 --no-open-browser
```

Gerekirse notebook tarafinda ayni klasoru acikca verebilirsin:

```python
url = launch_tree(
    df,
    target="risk_flag",
    start_server=False,
    open_browser=False,
    session_dir="/opt/app-root/src/interactive_decision_tree/.tree_sessions",
)
```

Jupyter/OpenShift proxy URL'sini elle cevirmek yerine dogrudan `base_url` verebilirsin:

```python
url = launch_tree(
    df,
    target="risk_flag",
    start_server=False,
    open_browser=False,
    base_url="https://<notebook-host>/notebook/<workspace>/proxy/8501/",
)
```

Route veya farkli host kullaniyorsan:

```python
url = launch_tree(
    df,
    target="risk_flag",
    start_server=False,
    open_browser=False,
    host="interactive-tree.apps.internal",
    scheme="https",
    port=8501,
)
```

`base_url` kullanmazsan query kismini elle tasirken `?data_id=...&work_id=...` degerlerini korumalisin.

### SQL

SQL modu Streamlit UI icinden SQLAlchemy URL ile tablo veya query sonucunu DataFrame'e cevirir. Sonuc `.tree_sessions/` altina snapshot olarak yazilir; sayfa yenilenince SQL tekrar calistirilmaz.

Oracle icin ornek:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[oracle]"
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

Notebook demo dosyasi `oracle_config/ora_config.ini` dosyasini da okuyabilir. Bu klasor git'e alinmaz. Beklenen format:

```ini
[ORA_PRD_ZTUSER]
host = ...
port = 1521
service_name = ...
user = ...
password = ...
```

## Ozellikler

- Leaf/node uzerinden manuel split secimi
- Candidate degiskenleri information gain'e gore siralama
- Information gain'i pozitif olmayan degiskenleri split secim listesinden gizleme
- Numeric, kategorik, kategorik target-profile grup splitleri
- Binary target icin kullanici tarafindan secilen positive class
- Default rate, AUC, Gini ve agac toplam gain metrikleri
- Optimal tree olusturup istedigin node'dan revize etme
- Notebook RAM'indeki pandas DataFrame'i `launch_tree(df)` ile lokal Streamlit app'e tasima
- SQLAlchemy ile SQL table/query sonucunu DataFrame olarak yukleyip agacta deneme
- Sidebar uzerinden Session DataFrame, CSV / Excel Upload, SQL ve Demo kaynaklari arasinda gecis
- UI uzerinden `.csv`, `.xlsx` ve `.xls` dosyalarini yukleyebilme
- Sayfa yenilense bile ayni `work_id` URL parametresiyle agaci otomatik geri yukleme
- Ayni data yukluyken exported tree JSON/pickle dosyasini UI'a import edip agaci editable halde devam ettirme
- Runnable nested tree JSON export
- Ayni Data Setup uzerinden `WOE Binning` workspace'ine gecip degisken bazli WOE mapping uretme
- Numeric/categorical initial binning, special/missing bin tanimi ve manuel WOE override
- Bin bazinda event/non-event/count/concentration/WOE/IV, degisken bazinda IV/Gini/monotonicity karsilastirmasi
- Butun degiskenler icin tek parca JSON, Excel, Python transformer ve SQL CASE export

## WOE Binning workflow

`Workspace = WOE Binning` secildiginde uygulama model kurmaz; degisken generation ekrani olarak calisir.

Akis:

1. `Data Setup` icinde train/test, target, positive class ve aktif degiskenleri uygula.
2. `WOE Binning` workspace'ine gec.
3. Sidebar'da WOE degiskenlerini ve global binning parametrelerini sec.
4. `Run initial WOE binning` ile her degiskenin ilk mapping'ini olustur.
5. Catalog ekraninda IV/Gini/monotonicity ile degiskenleri sirala.
6. Variable editor'da binleri, cutpoint/category gruplarini, special/missing politikasini ve `assigned_woe` degerlerini revize et.
7. Degisken status'unu `approved`, `rejected`, `needs_review` gibi isaretle.
8. Finalde tek parca WOE artifact indir.

WOE tarafinda her degisken icin `original_spec` ve `current_spec` ayri tutulur. Bu sayede ilk otomatik binning kaybolmaz; manuel revizyon sonrasi current IV/Gini, original IV/Gini ile karsilastirilir.

Manuel WOE mantigi:

```text
calculated_woe = event/non-event dagilimindan hesaplanan dogal WOE
assigned_woe   = kullanicinin elle verdigi override
export_woe     = assigned_woe varsa assigned_woe, yoksa calculated_woe
```

Special/missing destekleri:

- null/NaN missing
- blank string'i missing sayma opsiyonu
- `-999`, `-1`, `UNKNOWN`, `N/A` gibi degisken bazli special value listesi
- missing ve special binlere manuel WOE atama
- special bin'i protected tutma

Export dosyalari:

- `interactive_woe_mapping.json`: ana makine-okunur mapping contract
- `interactive_woe_report.xlsx`: Summary, Variable Metrics, Bin Details, Manual Edits raporu
- `woe_transformer.py`: pandas DataFrame'e `_WOE` kolonlari ekleyen transformer
- `woe_transform.sql`: SQL `CASE WHEN` mapping kodu

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

Tek musteri skorlamak icin:

```python
from interactive_decision_tree import load_tree_pickle, score_tree_payload

tree_payload = load_tree_pickle(r"C:\Users\Acer\Downloads\interactive_entropy_tree_runnable.pkl")

one_customer = {
    "age": 34,
    "income": 35_000,
    "tenure_months": 12,
    "segment": "C",
    "channel": "mobile",
    "region": "marmara",
}

score_result = score_tree_payload(tree_payload, one_customer)
score_result["prediction"]
score_result["prediction_probability"]
score_result["positive_class_probability"]
score_result["trace"]
```

`prediction` global bir risk threshold'una gore degil, musteri hangi leaf'e dustuyse o leaf'in cogunluk sinifina gore gelir. Binary target icin `positive_class_probability` leaf icindeki positive class oranidir; `prediction_probability` ise tahmin edilen sinifin leaf icindeki oranidir.

# 生成的 Neural HMM による単語クラスタリング

このリポジトリは、単語クラスタリングを教師なし品詞タグ付けとして実装するための最小プロジェクトです。  
モデルは Hidden Markov Model, HMM, を基本とし、潜在クラスタ `z_t` が観測単語 `x_t` を生成する生成モデルとして実装しています。

重要な設計条件として、emission は必ず次の向きです。

```text
p(x_t | z_t)
```

単語からタグを直接予測する `p(z_t | x_t)` 型の識別モデルではありません。  
実装上も `src/model.py` で emission を語彙方向に正規化しています。

```python
log_emission = torch.log_softmax(score, dim=1)
```

したがって `log_emission[k, v]` は `log p(x_t = v | z_t = k)` を表します。

## 現在の実装状況

実装済みの機能は以下です。

- CoNLL-U形式データの読み込み
- FORMとUPOSの読み込み
- 学習時はFORMのみ使用し、UPOSは評価専用として保持
- 訓練データからの語彙構築
- `<UNK>` による未知語処理
- 生成的HMM本体
- `initial_logits: [K]`
- `transition_logits: [K, K]`
- `emission_logits: [K, V]`
- log空間のforward algorithmによる `log p(x)` 計算
- 負の対数尤度による教師なし学習
- Viterbi algorithmによるクラスタ列デコード
- 特徴量付きemission
- prefix, suffix, capitalization, digit, hyphen, punctuation, length bucket
- 特徴量は `score(k, v)` の補正としてのみ使用
- many-to-one accuracy
- V-measure
- NMI
- ARI
- クラスタごとの頻出語とgold POS分布のTSV出力
- サンプルCoNLL-Uデータ
- UD English-EWT train/dev/testデータ
- tqdmによる学習時プログレスバー
- `uv` による環境構築手順

未実装または今後の拡張候補は以下です。

- padding付きミニバッチ
- UD English-EWTなど実データの同梱
- 複数seed・複数Kの一括実験スクリプト
- 可視化
- レポート用の実験表生成

## モデル概要

文を単語列 `x = x_1, ..., x_T`、潜在クラスタ列を `z = z_1, ..., z_T` とします。

同時確率は次の通りです。

```text
p(x, z) = p(z_1) p(x_1 | z_1)
          prod_{t=2..T} p(z_t | z_{t-1}) p(x_t | z_t)
```

観測されるのは単語列 `x` のみなので、潜在状態列 `z` はforward algorithmで周辺化します。

```text
p(x) = sum_z p(x, z)
```

学習では、文ごとの負の対数尤度を最小化します。

```text
loss = -log p(x)
```

## 特徴量付きemission

特徴量を使う場合も、モデルの向きは `p(x | z)` のままです。

```text
score(k, v) = base_emission_logits(k, v)
              + sum_{f in F(v)} feature_weights(f, k)

p(x = v | z = k) = softmax_v score(k, v)
```

禁止している形は次です。

```text
MLP(features(x)) -> p(z | x)
```

この実装では、特徴量はクラスタごとの語彙生成スコアを補正するためにだけ使います。

## フォルダ構成

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── sample.conllu
│   ├── en_ewt-ud-train.conllu
│   ├── en_ewt-ud-dev.conllu
│   └── en_ewt-ud-test.conllu
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── decode.py
│   ├── evaluate.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── outputs/
    └── 実行時に生成
```

各ファイルの役割は以下です。

```text
src/data.py       CoNLL-U読み込み、語彙作成、Dataset
src/features.py   単語特徴量抽出、特徴ID化
src/model.py      GenerativeHMM、forward algorithm、Viterbi
src/train.py      教師なし学習CLI
src/decode.py     学習済みモデルによるViterbiデコードCLI
src/evaluate.py   many-to-one accuracyなどの評価CLI
src/utils.py      seed固定、JSON保存、device取得
```

`outputs/` と `.venv/` は `.gitignore` に入れています。`data/` はGit管理対象として残しています。

## Git管理方針

Gitに入れるもの:

- `src/` の実装
- `README.md`
- `requirements.txt`
- `.gitignore`
- `data/sample.conllu`
- `data/en_ewt-ud-*.conllu`

Gitから除外するもの:

- `.venv/`
- `outputs/`
- Pythonキャッシュ
- pytest, mypy, ruffなどのツールキャッシュ
- build/dist系の生成物

## データ

現在、`data/` には以下を置いています。

```text
data/sample.conllu              動作確認用の小さいサンプル
data/en_ewt-ud-train.conllu     UD English-EWT train
data/en_ewt-ud-dev.conllu       UD English-EWT dev
data/en_ewt-ud-test.conllu      UD English-EWT test
```

UD English-EWTはGitHubのUniversal Dependencies公式リポジトリから取得しています。

既存readerで確認した件数は以下です。

```text
train: 12544 sentences, 204578 tokens
dev:    2001 sentences,  25148 tokens
test:   2077 sentences,  25094 tokens
```

`min_freq=2` でtrainから語彙を作ると、語彙サイズは `<UNK>` を含めて9,874です。

## 環境

推奨は `uv` です。

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt
```

この環境では、検証時に以下が入りました。

```text
Python: .venv/bin/python
torch: 2.12.1
numpy: 2.5.0
tqdm: プログレスバー表示用
```

注意点として、Anacondaなど別のPython環境がある場合、単に `uv pip install -r requirements.txt` と実行するとプロジェクトの `.venv` ではなく別環境に入ることがあります。  
そのため、このリポジトリでは次のように `--python` を明示する運用にしています。

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

標準の `venv` と `pip` を使う場合は以下です。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 実行方法

### 最小HMMの学習

学習中は `tqdm` のプログレスバーで、epoch内のbatch進捗と暫定 `nll_per_token` を確認できます。  
表示を消したい場合は `--no-progress` を付けます。

UD English-EWTで学習する例です。

```bash
./.venv/bin/python -m src.train \
  --train data/en_ewt-ud-train.conllu \
  --dev data/en_ewt-ud-dev.conllu \
  --num-tags 17 \
  --min-freq 2 \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.01 \
  --output-dir outputs/ewt_run1
```

プログレスバーを無効化する例です。

```bash
./.venv/bin/python -m src.train \
  --train data/en_ewt-ud-train.conllu \
  --dev data/en_ewt-ud-dev.conllu \
  --num-tags 17 \
  --min-freq 2 \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.01 \
  --no-progress \
  --output-dir outputs/ewt_run1
```

小さいサンプルで学習する例です。

```bash
./.venv/bin/python -m src.train \
  --train data/sample.conllu \
  --dev data/sample.conllu \
  --num-tags 5 \
  --min-freq 1 \
  --epochs 10 \
  --batch-size 1 \
  --lr 0.05 \
  --output-dir outputs/sample
```

### 特徴量付きHMMの学習

```bash
./.venv/bin/python -m src.train \
  --train data/sample.conllu \
  --dev data/sample.conllu \
  --num-tags 5 \
  --min-freq 1 \
  --use-features \
  --epochs 10 \
  --batch-size 1 \
  --lr 0.05 \
  --output-dir outputs/sample_features
```

### デコード

```bash
./.venv/bin/python -m src.decode \
  --model outputs/ewt_run1/model.pt \
  --input data/en_ewt-ud-test.conllu \
  --output outputs/ewt_run1/predictions.conllu
```

予測クラスタは、出力CoNLL-UのUPOS列に `CLUSTER_0`, `CLUSTER_1` のように保存します。

### 評価

```bash
./.venv/bin/python -m src.evaluate \
  --gold data/en_ewt-ud-test.conllu \
  --pred outputs/ewt_run1/predictions.conllu \
  --output outputs/ewt_run1/metrics.json
```

評価時には `metrics.json` に加えて、同じディレクトリに `cluster_summary.tsv` も出力します。

## 出力ファイル

学習・デコード・評価後、典型的には以下が生成されます。

```text
outputs/sample/
├── model.pt
├── config.json
├── vocab.json
├── train_log.jsonl
├── predictions.conllu
├── metrics.json
└── cluster_summary.tsv
```

`model.pt` には以下を保存しています。

- model state dict
- config
- vocab
- feature_to_id
- word_feature_ids

## 現在の実験結果

UD English-EWTで、最小HMMを以下の条件で学習しました。

```bash
./.venv/bin/python -m src.train \
  --train data/en_ewt-ud-train.conllu \
  --dev data/en_ewt-ud-dev.conllu \
  --num-tags 17 \
  --min-freq 2 \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.01 \
  --output-dir outputs/ewt_run1
```

学習後、test splitに対してViterbiデコードと評価を行いました。

```bash
./.venv/bin/python -m src.decode \
  --model outputs/ewt_run1/model.pt \
  --input data/en_ewt-ud-test.conllu \
  --output outputs/ewt_run1/predictions.conllu

./.venv/bin/python -m src.evaluate \
  --gold data/en_ewt-ud-test.conllu \
  --pred outputs/ewt_run1/predictions.conllu \
  --output outputs/ewt_run1/metrics.json
```

学習結果:

```text
best_epoch: 30
train_nll_per_token: 6.1556
dev_nll_per_token: 5.9404
vocab_size: 9874
num_tags: 17
```

test評価:

```json
{
  "many_to_one_accuracy": 0.3043,
  "v_measure": 0.1905,
  "nmi": 0.1905,
  "ari": 0.0921,
  "num_clusters_used": 14,
  "num_tokens": 25094
}
```

注意: `outputs/` は `.gitignore` 対象なので、上記の学習済みモデルや評価ファイルは通常のGit管理には含めません。必要な場合は同じコマンドで再生成します。

### test出力例

例1:

```text
INPUT:
What if Google Morphed Into GoogleOS ?

GOLD:
What/PRON if/SCONJ Google/PROPN Morphed/VERB Into/ADP GoogleOS/PROPN ?/PUNCT

PRED:
What/C8(PRON) if/C11(PRON) Google/C10(NOUN) Morphed/C3(PUNCT) Into/C15(PROPN) GoogleOS/C15(PROPN) ?/C15(PROPN)
```

例2:

```text
INPUT:
What if Google expanded on its search - engine ( and now e-mail ) wares into a full - fledged operating system ?

GOLD:
What/PRON if/SCONJ Google/PROPN expanded/VERB on/ADP its/PRON search/NOUN -/PUNCT engine/NOUN (/PUNCT and/CCONJ now/ADV e-mail/NOUN )/PUNCT wares/NOUN into/ADP a/DET full/ADV -/PUNCT fledged/ADJ operating/NOUN system/NOUN ?/PUNCT

PRED:
What/C8(PRON) if/C11(PRON) Google/C10(NOUN) expanded/C10(NOUN) on/C10(NOUN) its/C10(NOUN) search/C10(NOUN) -/C10(NOUN) engine/C10(NOUN) (/C10(NOUN) and/C10(NOUN) now/C10(NOUN) e-mail/C3(PUNCT) )/C14(PUNCT) wares/C10(NOUN) into/C10(NOUN) a/C10(NOUN) full/C10(NOUN) -/C10(NOUN) fledged/C10(NOUN) operating/C10(NOUN) system/C10(NOUN) ?/C3(PUNCT)
```

`C8(PRON)` のような表記は、クラスタID `8` がmany-to-one評価で `PRON` に対応したことを示しています。モデル自体が出力しているのは `CLUSTER_8` のようなクラスタIDであり、括弧内の品詞名は評価後の解釈です。

現状の観察:

- 固有名詞らしい語は `C15(PROPN)` に寄る傾向があります。
- `C10(NOUN)` が大きなクラスタになっており、名詞だけでなく前置詞、限定詞、句読点も吸っています。
- 句読点クラスタは一部まとまりますが、完全には分離していません。
- 教師なしHMMとしては一通り動いていますが、クラスタ分離はまだ粗いです。

次の改善候補:

- `--use-features` を付けた特徴量付きemissionで本学習する
- `K=12`, `17`, `20` を比較する
- `seed=0`, `1`, `2` を比較する
- 学習率を下げる
- epoch数を増やすか、dev NLLに基づくearly stoppingを追加する

## 動作確認済みコマンド

以下のスモークテストは確認済みです。

```bash
./.venv/bin/python -m compileall src

./.venv/bin/python -m src.train \
  --train data/sample.conllu \
  --dev data/sample.conllu \
  --num-tags 5 \
  --min-freq 1 \
  --epochs 1 \
  --batch-size 1 \
  --lr 0.05 \
  --output-dir outputs/smoke_final

./.venv/bin/python -m src.decode \
  --model outputs/smoke_final/model.pt \
  --input data/sample.conllu \
  --output outputs/smoke_final/predictions.conllu

./.venv/bin/python -m src.evaluate \
  --gold data/sample.conllu \
  --pred outputs/smoke_final/predictions.conllu \
  --output outputs/smoke_final/metrics.json
```

特徴量付きemissionについても、短い学習実行は確認済みです。

```bash
./.venv/bin/python -m src.train \
  --train data/sample.conllu \
  --dev data/sample.conllu \
  --num-tags 5 \
  --min-freq 1 \
  --use-features \
  --epochs 1 \
  --batch-size 1 \
  --lr 0.05 \
  --output-dir outputs/smoke_features_final
```

UD English-EWTのtrain/dev/testでも、フォーマット互換性の確認として次の流れが通っています。

```bash
./.venv/bin/python -m src.train \
  --train data/en_ewt-ud-train.conllu \
  --dev data/en_ewt-ud-dev.conllu \
  --num-tags 5 \
  --min-freq 2 \
  --epochs 1 \
  --batch-size 32 \
  --lr 0.01 \
  --output-dir outputs/ewt_format_check

./.venv/bin/python -m src.decode \
  --model outputs/ewt_format_check/model.pt \
  --input data/en_ewt-ud-test.conllu \
  --output outputs/ewt_format_check/predictions.conllu

./.venv/bin/python -m src.evaluate \
  --gold data/en_ewt-ud-test.conllu \
  --pred outputs/ewt_format_check/predictions.conllu \
  --output outputs/ewt_format_check/metrics.json
```

この確認実行は1 epochかつ `num-tags=5` のフォーマット確認用なので、評価値は実験結果として解釈するものではありません。

## 技術メモ

- 学習時にgold UPOSは損失計算へ渡していません。
- `src/train.py` は `word_ids` のみをモデルに渡します。
- `src/evaluate.py` だけがgold UPOSを使います。
- `src/train.py` は `tqdm` でtrain/devの進捗を表示します。
- `--no-progress` を付けるとプログレスバーを無効化できます。
- forward algorithmとViterbi algorithmはlog空間で実装しています。
- `torch.logsumexp` を使って数値安定性を確保しています。
- 初期実装では `batch-size=1` を主想定にしています。
- `batch-size > 1` でも文ごとにforwardを計算して平均lossを取るため動きます。
- 同一batch内では `log_initial`, `log_transition`, `log_emission` を使い回すようにしており、実データでも短い確認学習は回せます。
- paddingによる完全なベクトル化は未実装です。
- 実データでは `--num-tags 12`, `17`, `20` と `--seed 0`, `1`, `2` の比較が次の実験候補です。

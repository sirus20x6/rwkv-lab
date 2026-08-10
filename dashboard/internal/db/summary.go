package db

import (
	"database/sql"
	"fmt"
	"log"
)

const runSummarySelect = `SELECT r.id, r.name, COALESCE(r.last_update_ts,0), COALESCE(r.tags_json,'[]'),
	COALESCE(x.n_train,0), COALESCE(x.n_eval,0), COALESCE(x.n_ckpt,0),
	x.latest_train_step, x.latest_train_loss, x.latest_eval_step,
	x.latest_eval_ppl, x.latest_eval_top1, x.best_ppl, x.best_top1,
	(SELECT e.step FROM eval_events e WHERE e.run_id=r.id AND e.ppl IS NOT NULL
	 ORDER BY e.ppl ASC, e.step ASC LIMIT 1),
	COALESCE(x.has_horizons,0)
	FROM runs r LEFT JOIN run_rollups x ON x.run_id=r.id`

type rowScanner interface {
	Scan(dest ...any) error
}

// RunSummary is one row in the sidebar run list.
type RunSummary struct {
	ID           int64    `json:"-"`
	Name         string   `json:"name"`
	LastUpdateTs float64  `json:"last_update_ts"`
	LatestStep   *int64   `json:"latest_step"`
	LatestLoss   *float64 `json:"latest_loss"`
	LatestPPL    *float64 `json:"latest_ppl"`
	LatestTop1   *float64 `json:"latest_top1"`
	BestPPL      *float64 `json:"best_ppl"`
	BestPPLStep  *int64   `json:"best_ppl_step"`
	BestTop1     *float64 `json:"best_top1"`
	NTrain       int      `json:"n_train"`
	NEval        int      `json:"n_eval"`
	NCkpt        int      `json:"n_ckpt"`
	HasHorizons  bool     `json:"has_horizons"`
	Status       string   `json:"status"` // healthy|stalling|cold (by log age; proc may promote)
	TagsJSON     string   `json:"-"`      // raw tags_json column ("[]" when unset)
}

// RunSummaries returns every run with its ingestion-time rollup in one O(runs)
// query. Status here is purely log-age derived; the caller can promote a run to
// "healthy" when a live process is attached.
func (d *DB) RunSummaries(nowTs float64) ([]RunSummary, error) {
	rows, err := d.Query(runSummarySelect)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var order []*RunSummary
	for rows.Next() {
		s, err := scanRunSummary(rows, nowTs)
		if err != nil {
			return nil, err
		}
		order = append(order, &s)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	out := make([]RunSummary, 0, len(order))
	for _, s := range order {
		out = append(out, *s)
	}
	return out, nil
}

// RunSummaryByName returns one run's rollup through the unique runs.name index.
// It is the per-browser live-refresh path, so it must not scan every run once
// per browser per second like RunSummaries does for the shared tick cache.
func (d *DB) RunSummaryByName(name string, nowTs float64) (RunSummary, bool, error) {
	s, err := scanRunSummary(d.QueryRow(runSummarySelect+` WHERE r.name=?`, name), nowTs)
	if err == sql.ErrNoRows {
		return RunSummary{}, false, nil
	}
	if err != nil {
		return RunSummary{}, false, err
	}
	return s, true, nil
}

func scanRunSummary(row rowScanner, nowTs float64) (RunSummary, error) {
	s := RunSummary{}
	var trainStep, evalStep, bestPPLStep sql.NullInt64
	var trainLoss, evalPPL, evalTop1, bestPPL, bestTop1 sql.NullFloat64
	var hasHorizons int
	if err := row.Scan(&s.ID, &s.Name, &s.LastUpdateTs, &s.TagsJSON,
		&s.NTrain, &s.NEval, &s.NCkpt, &trainStep, &trainLoss, &evalStep,
		&evalPPL, &evalTop1, &bestPPL, &bestTop1, &bestPPLStep, &hasHorizons); err != nil {
		return RunSummary{}, err
	}
	if trainStep.Valid {
		v := trainStep.Int64
		s.LatestStep = &v
	}
	if trainLoss.Valid {
		v := trainLoss.Float64
		s.LatestLoss = &v
	}
	if evalStep.Valid && (s.LatestStep == nil || evalStep.Int64 > *s.LatestStep) {
		v := evalStep.Int64
		s.LatestStep = &v
	}
	if evalPPL.Valid {
		v := evalPPL.Float64
		s.LatestPPL = &v
	}
	if evalTop1.Valid {
		v := evalTop1.Float64
		s.LatestTop1 = &v
	}
	if bestPPL.Valid {
		v := bestPPL.Float64
		s.BestPPL = &v
	}
	if bestPPLStep.Valid {
		v := bestPPLStep.Int64
		s.BestPPLStep = &v
	}
	if bestTop1.Valid {
		v := bestTop1.Float64
		s.BestTop1 = &v
	}
	s.HasHorizons = hasHorizons != 0
	switch age := nowTs - s.LastUpdateTs; {
	case age < 300:
		s.Status = "healthy"
	case age < 900:
		s.Status = "stalling"
	default:
		s.Status = "cold"
	}
	return s, nil
}

// LatestCodecRelByRun returns the newest non-null codec metric for every run in
// one query. The conversion board calls this once per shared tick instead of
// issuing KPI/count/stat queries independently for each of 32 layers.
func (d *DB) LatestCodecRelByRun() (map[string]*float64, error) {
	rows, err := d.Query(`SELECT r.name,
		(SELECT json_extract(t.extra_json,'$.codec_rel')
		 FROM train_events t INDEXED BY idx_train_codec_rel WHERE t.run_id=r.id
		   AND json_extract(t.extra_json,'$.codec_rel') IS NOT NULL
		 ORDER BY t.step DESC LIMIT 1)
		FROM runs r`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]*float64{}
	for rows.Next() {
		var name string
		var value sql.NullFloat64
		if err := rows.Scan(&name, &value); err != nil {
			return nil, err
		}
		if value.Valid {
			v := value.Float64
			out[name] = &v
		}
	}
	return out, rows.Err()
}

func (d *DB) eachCount(query string, byID map[int64]*RunSummary, set func(*RunSummary, int)) error {
	rows, err := d.Query(query)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var rid int64
		var n int
		if err := rows.Scan(&rid, &n); err != nil {
			return err
		}
		if s := byID[rid]; s != nil {
			set(s, n)
		}
	}
	return rows.Err()
}

func (d *DB) scanLatestTrain(byID map[int64]*RunSummary) error {
	// max(step) GROUP BY walks the narrow (run_id,step) index; the join then does
	// one point lookup per run for the payload. The previous window-function form
	// read every full row (incl. extra_json) — ~0.5s/call on a 50MB DB.
	rows, err := d.Query(`SELECT t.run_id, t.step, t.loss FROM train_events t
		JOIN (SELECT run_id, max(step) AS m FROM train_events GROUP BY run_id) x
		  ON t.run_id = x.run_id AND t.step = x.m`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var rid, step int64
		var loss sql.NullFloat64
		if err := rows.Scan(&rid, &step, &loss); err != nil {
			return err
		}
		if s := byID[rid]; s != nil {
			st := step
			s.LatestStep = &st
			if loss.Valid {
				v := loss.Float64
				s.LatestLoss = &v
			}
		}
	}
	return rows.Err()
}

func (d *DB) scanLatestEval(byID map[int64]*RunSummary) error {
	rows, err := d.Query(`SELECT e.run_id, e.step, e.ppl, e.top1 FROM eval_events e
		JOIN (SELECT run_id, max(step) AS m FROM eval_events GROUP BY run_id) x
		  ON e.run_id = x.run_id AND e.step = x.m`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var rid, step int64
		var ppl, top1 sql.NullFloat64
		if err := rows.Scan(&rid, &step, &ppl, &top1); err != nil {
			return err
		}
		s := byID[rid]
		if s == nil {
			continue
		}
		// promote latest step if eval is ahead of train
		if s.LatestStep == nil || step > *s.LatestStep {
			st := step
			s.LatestStep = &st
		}
		if ppl.Valid {
			v := ppl.Float64
			s.LatestPPL = &v
		}
		if top1.Valid {
			v := top1.Float64
			s.LatestTop1 = &v
		}
	}
	return rows.Err()
}

func (d *DB) scanBestEval(byID map[int64]*RunSummary) error {
	rows, err := d.Query(`SELECT run_id, min(ppl), max(top1) FROM eval_events GROUP BY run_id`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var rid int64
		var ppl, top1 sql.NullFloat64
		if err := rows.Scan(&rid, &ppl, &top1); err != nil {
			return err
		}
		s := byID[rid]
		if s == nil {
			continue
		}
		if ppl.Valid {
			v := ppl.Float64
			s.BestPPL = &v
		}
		if top1.Valid {
			v := top1.Float64
			s.BestTop1 = &v
		}
	}
	return rows.Err()
}

// RunKPIs is the selected-run KPI strip payload.
type RunKPIs struct {
	Step                           *int64   `json:"step"`
	Loss                           *float64 `json:"loss"`
	PPL                            *float64 `json:"ppl"`
	CaptionPPL                     *float64 `json:"caption_ppl"`
	OCRPPL                         *float64 `json:"ocr_ppl"`
	OCRShuffledPPL                 *float64 `json:"ocr_shuffled_ppl"`
	OCRConditioningNLL             *float64 `json:"ocr_image_conditioning_nll_delta"`
	OCRConditioningExampleCoverage *float64 `json:"ocr_image_conditioning_example_coverage"`
	OCRConditioningTokenCoverage   *float64 `json:"ocr_image_conditioning_token_coverage"`
	StructuredPPL                  *float64 `json:"structured_ppl"`
	StructuredCoordPPL             *float64 `json:"structured_coordinate_ppl"`
	// StructuredBoxIoU is the instance-weighted true IoU, emitted under the
	// current key structured_box_iou_instance.
	StructuredBoxIoU *float64 `json:"structured_box_iou_instance"`
	// StructuredBoxIoULegacy carries the RETIRED structured_box_iou key, which
	// held a differently-weighted quantity. Rows written before the rename still
	// have only that key; they are surfaced as an explicitly-labelled legacy
	// series rather than silently rendered as if they were the new metric, and
	// never as the new field (mixing the two would make a run's history lie).
	StructuredBoxIoULegacy *float64 `json:"structured_box_iou_legacy"`
	StructuredBoxGIoU      *float64 `json:"structured_box_giou"`
	StructuredMaskDice     *float64 `json:"structured_mask_dice"`
	BestPPL                *float64 `json:"best_ppl"`
	BestPPLStep            *int64   `json:"best_ppl_step"`
	Top1                   *float64 `json:"top1"`
	BestTop1               *float64 `json:"best_top1"`
	BestTop1Step           *int64   `json:"best_top1_step"`
	BestLoss               *float64 `json:"best_loss"`
	BestLossStep           *int64   `json:"best_loss_step"`
	Toks                   *float64 `json:"toks"`
	LR                     *float64 `json:"lr"`
	Gnorm                  *float64 `json:"gnorm"`
	NTrain                 int      `json:"n_train"`
	NEval                  int      `json:"n_eval"`
	NCkpt                  int      `json:"n_ckpt"`
}

// kpiScan normalizes one KPI QueryRow result. A run with no rows of that kind
// yet is ordinary (the KPI renders "—"), but every other error is real: these
// SELECTs carry a dozen-plus json_extract columns whose count must match their
// Scan list exactly, and a silently discarded error there makes EVERY KPI in
// the strip render "—" with nothing in the log to explain it.
func kpiScan(what, run string, err error) error {
	if err == nil || err == sql.ErrNoRows {
		return nil
	}
	log.Printf("[kpi] run %q %s: %v", run, what, err)
	return fmt.Errorf("run %q %s KPI: %w", run, what, err)
}

// RunKPIsByName computes the KPI strip for one run (a few quick single-run queries).
func (d *DB) RunKPIsByName(name string) (RunKPIs, bool, error) {
	var rid int64
	err := d.QueryRow(`SELECT id FROM runs WHERE name=?`, name).Scan(&rid)
	if err == sql.ErrNoRows {
		return RunKPIs{}, false, nil
	}
	if err != nil {
		return RunKPIs{}, false, err
	}
	k := RunKPIs{}

	// latest train: step, loss, tok_per_sec, lr, gnorm
	var step sql.NullInt64
	var loss, toks, lr, gnorm sql.NullFloat64
	if err := kpiScan("latest train", name, d.QueryRow(
		`SELECT step, loss, tok_per_sec, lr, gnorm FROM train_events WHERE run_id=? ORDER BY step DESC LIMIT 1`, rid).
		Scan(&step, &loss, &toks, &lr, &gnorm)); err != nil {
		return RunKPIs{}, true, err
	}
	if step.Valid {
		k.Step = &step.Int64
	}
	if loss.Valid {
		k.Loss = &loss.Float64
	}
	if toks.Valid {
		k.Toks = &toks.Float64
	}
	if lr.Valid {
		k.LR = &lr.Float64
	}
	if gnorm.Valid {
		k.Gnorm = &gnorm.Float64
	}

	// latest eval: ppl, top1 (and promote step)
	var estep sql.NullInt64
	var ppl, top1, captionPPL, ocrPPL, ocrShuffledPPL, ocrConditioningNLL sql.NullFloat64
	var ocrConditioningExampleCoverage, ocrConditioningTokenCoverage sql.NullFloat64
	var structuredPPL, structuredCoordPPL sql.NullFloat64
	var structuredBoxIoU, structuredBoxIoULegacy sql.NullFloat64
	var structuredBoxGIoU, structuredMaskDice sql.NullFloat64
	if err := kpiScan("latest eval", name, d.QueryRow(`SELECT step, ppl, top1,
		json_extract(extra_json,'$.caption_ppl'),
		json_extract(extra_json,'$.ocr_ppl'),
		json_extract(extra_json,'$.ocr_shuffled_ppl'),
		json_extract(extra_json,'$.ocr_image_conditioning_nll_delta'),
		json_extract(extra_json,'$.ocr_image_conditioning_example_coverage'),
		json_extract(extra_json,'$.ocr_image_conditioning_token_coverage'),
		json_extract(extra_json,'$.structured_ppl'),
		json_extract(extra_json,'$.structured_coordinate_ppl'),
		json_extract(extra_json,'$.structured_box_iou_instance'),
		json_extract(extra_json,'$.structured_box_iou'),
		json_extract(extra_json,'$.structured_box_giou'),
		json_extract(extra_json,'$.structured_mask_dice')
		FROM eval_events WHERE run_id=? ORDER BY step DESC LIMIT 1`, rid).
		Scan(&estep, &ppl, &top1, &captionPPL, &ocrPPL,
			&ocrShuffledPPL, &ocrConditioningNLL,
			&ocrConditioningExampleCoverage, &ocrConditioningTokenCoverage,
			&structuredPPL,
			&structuredCoordPPL, &structuredBoxIoU, &structuredBoxIoULegacy,
			&structuredBoxGIoU, &structuredMaskDice)); err != nil {
		return RunKPIs{}, true, err
	}
	if estep.Valid && (k.Step == nil || estep.Int64 > *k.Step) {
		k.Step = &estep.Int64
	}
	if ppl.Valid {
		k.PPL = &ppl.Float64
	}
	if top1.Valid {
		k.Top1 = &top1.Float64
	}
	if captionPPL.Valid {
		k.CaptionPPL = &captionPPL.Float64
	}
	if ocrPPL.Valid {
		k.OCRPPL = &ocrPPL.Float64
	}
	if ocrShuffledPPL.Valid {
		k.OCRShuffledPPL = &ocrShuffledPPL.Float64
	}
	if ocrConditioningNLL.Valid {
		k.OCRConditioningNLL = &ocrConditioningNLL.Float64
	}
	if ocrConditioningExampleCoverage.Valid {
		k.OCRConditioningExampleCoverage = &ocrConditioningExampleCoverage.Float64
	}
	if ocrConditioningTokenCoverage.Valid {
		k.OCRConditioningTokenCoverage = &ocrConditioningTokenCoverage.Float64
	}
	if structuredPPL.Valid {
		k.StructuredPPL = &structuredPPL.Float64
	}
	if structuredCoordPPL.Valid {
		k.StructuredCoordPPL = &structuredCoordPPL.Float64
	}
	if structuredBoxIoU.Valid {
		k.StructuredBoxIoU = &structuredBoxIoU.Float64
	} else if structuredBoxIoULegacy.Valid {
		// Pre-rename row: the only value it has is the retired metric.
		k.StructuredBoxIoULegacy = &structuredBoxIoULegacy.Float64
	}
	if structuredBoxGIoU.Valid {
		k.StructuredBoxGIoU = &structuredBoxGIoU.Float64
	}
	if structuredMaskDice.Valid {
		k.StructuredMaskDice = &structuredMaskDice.Float64
	}

	// best ppl (min) + its step
	var bppl sql.NullFloat64
	var bpstep sql.NullInt64
	if err := kpiScan("best eval ppl", name, d.QueryRow(
		`SELECT ppl, step FROM eval_events WHERE run_id=? AND ppl IS NOT NULL ORDER BY ppl ASC LIMIT 1`, rid).
		Scan(&bppl, &bpstep)); err != nil {
		return RunKPIs{}, true, err
	}
	if bppl.Valid {
		k.BestPPL = &bppl.Float64
	}
	if bpstep.Valid {
		k.BestPPLStep = &bpstep.Int64
	}

	// best top1 (max) + its step
	var btop1 sql.NullFloat64
	var bt1step sql.NullInt64
	if err := kpiScan("best eval top1", name, d.QueryRow(
		`SELECT top1, step FROM eval_events WHERE run_id=? AND top1 IS NOT NULL ORDER BY top1 DESC LIMIT 1`, rid).
		Scan(&btop1, &bt1step)); err != nil {
		return RunKPIs{}, true, err
	}
	if btop1.Valid {
		k.BestTop1 = &btop1.Float64
	}
	if bt1step.Valid {
		k.BestTop1Step = &bt1step.Int64
	}
	// best (min) train loss + its step
	var bloss sql.NullFloat64
	var blstep sql.NullInt64
	if err := kpiScan("best train loss", name, d.QueryRow(
		`SELECT loss, step FROM train_events WHERE run_id=? AND loss IS NOT NULL ORDER BY loss ASC LIMIT 1`, rid).
		Scan(&bloss, &blstep)); err != nil {
		return RunKPIs{}, true, err
	}
	if bloss.Valid {
		k.BestLoss = &bloss.Float64
	}
	if blstep.Valid {
		k.BestLossStep = &blstep.Int64
	}

	k.NTrain, k.NEval, k.NCkpt, err = d.EventCounts(rid)
	if err != nil {
		return k, true, err
	}
	return k, true, nil
}

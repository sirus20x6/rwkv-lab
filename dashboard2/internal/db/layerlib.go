package db

// AcceptLayer records a layer-promotion candidate (provenance). It does not
// touch converted_layers_lib — promotion is a deliberate user step via
// assemble_looped.py. Upserts on layer.
func (d *DB) AcceptLayer(layer int, run string, srcStep int64, libPath string, ppl *float64, ts float64) error {
	_, err := d.Exec(
		`INSERT INTO layer_lib(layer, run_name, src_step, lib_path, ppl, codec_rel, accepted_ts)
		 VALUES(?,?,?,?,?,NULL,?)
		 ON CONFLICT(layer) DO UPDATE SET run_name=excluded.run_name, src_step=excluded.src_step,
		   lib_path=excluded.lib_path, ppl=excluded.ppl, accepted_ts=excluded.accepted_ts`,
		layer, run, srcStep, libPath, ppl, ts)
	return err
}

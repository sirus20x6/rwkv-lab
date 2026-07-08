package db

import "database/sql"

// QueueItem is one entry in the launch queue.
type QueueItem struct {
	ID         int64   `json:"id"`
	Script     string  `json:"script"`
	Args       string  `json:"args"`
	Status     string  `json:"status"` // queued|running|done|failed|canceled
	Priority   int     `json:"priority"`
	PID        *int64  `json:"pid"`
	EnqueuedTs float64 `json:"enqueued_ts"`
	LogPath    string  `json:"log_path"`
}

// Enqueue appends a run to the queue.
func (d *DB) Enqueue(script, args string, priority int, ts float64) (int64, error) {
	res, err := d.Exec(
		`INSERT INTO launch_queue(enqueued_ts,script,args,status,priority) VALUES(?,?,?,'queued',?)`,
		ts, script, args, priority)
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

// ActiveQueue returns queued + running items (newest activity first within status).
func (d *DB) ActiveQueue() ([]QueueItem, error) {
	rows, err := d.Query(
		`SELECT id,script,args,status,priority,pid,COALESCE(enqueued_ts,0),COALESCE(log_path,'')
		 FROM launch_queue WHERE status IN ('queued','running')
		 ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, priority DESC, id ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanQueue(rows)
}

// NextQueued returns the highest-priority queued item.
func (d *DB) NextQueued() (QueueItem, bool, error) {
	rows, err := d.Query(
		`SELECT id,script,args,status,priority,pid,COALESCE(enqueued_ts,0),COALESCE(log_path,'')
		 FROM launch_queue WHERE status='queued' ORDER BY priority DESC, id ASC LIMIT 1`)
	if err != nil {
		return QueueItem{}, false, err
	}
	defer rows.Close()
	items, err := scanQueue(rows)
	if err != nil || len(items) == 0 {
		return QueueItem{}, false, err
	}
	return items[0], true, nil
}

// RunningQueue returns items currently marked running (for PID reconciliation).
func (d *DB) RunningQueue() ([]QueueItem, error) {
	rows, err := d.Query(
		`SELECT id,script,args,status,priority,pid,COALESCE(enqueued_ts,0),COALESCE(log_path,'')
		 FROM launch_queue WHERE status='running'`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanQueue(rows)
}

func (d *DB) MarkRunning(id int64, pid int, logPath string, ts float64) error {
	_, err := d.Exec(`UPDATE launch_queue SET status='running', pid=?, started_ts=?, log_path=? WHERE id=?`,
		pid, ts, logPath, id)
	return err
}

func (d *DB) MarkFinished(id int64, status string, ts float64) error {
	_, err := d.Exec(`UPDATE launch_queue SET status=?, finished_ts=? WHERE id=?`, status, ts, id)
	return err
}

// CancelQueued cancels an item only if it is still queued (never a running one).
func (d *DB) CancelQueued(id int64) (bool, error) {
	res, err := d.Exec(`UPDATE launch_queue SET status='canceled' WHERE id=? AND status='queued'`, id)
	if err != nil {
		return false, err
	}
	n, _ := res.RowsAffected()
	return n > 0, nil
}

func scanQueue(rows *sql.Rows) ([]QueueItem, error) {
	var out []QueueItem
	for rows.Next() {
		var q QueueItem
		var pid sql.NullInt64
		if err := rows.Scan(&q.ID, &q.Script, &q.Args, &q.Status, &q.Priority, &pid, &q.EnqueuedTs, &q.LogPath); err != nil {
			return nil, err
		}
		if pid.Valid {
			q.PID = &pid.Int64
		}
		out = append(out, q)
	}
	return out, rows.Err()
}

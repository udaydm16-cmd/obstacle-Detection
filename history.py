"""Local SQLite storage for measured passages and review events."""
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

DATABASE = Path(__file__).resolve().parent / 'data' / 'history.sqlite3'
WINDOW = 72 * 3600


def site_key(source, road, gates=None):
    setup = json.dumps([source, road, gates], separators=(',', ':'))
    return 'camera-' + hashlib.sha256(setup.encode()).hexdigest()[:16]


class History:
    def __init__(self, path=DATABASE):
        self.path = path
        with closing(self.connect()) as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS passages (
                    site TEXT, occurred REAL, lane INTEGER, slow INTEGER, merged INTEGER
                );
                CREATE INDEX IF NOT EXISTS passages_site_time ON passages(site, occurred);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, occurred REAL, source TEXT, site TEXT,
                    kind TEXT, reason TEXT, evidence TEXT, lane INTEGER
                );
                CREATE INDEX IF NOT EXISTS events_site_time ON events(site, occurred);
            ''')

    def calibration(self, camera, corners=None):
        with closing(self.connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS calibration (camera INTEGER PRIMARY KEY, corners TEXT)')
            if corners is not None:
                db.execute('INSERT OR REPLACE INTO calibration VALUES (?,?)', (camera,json.dumps(corners)))
            row = db.execute('SELECT corners FROM calibration WHERE camera=?',(camera,)).fetchone()
            return json.loads(row['corners']) if row else None

    def connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def restore(self, site, now):
        with closing(self.connect()) as db, db:
            db.execute('DELETE FROM passages WHERE occurred < ?', (now-WINDOW,))
            rows = db.execute('SELECT occurred,lane,slow,merged FROM passages '
                              'WHERE site=? AND occurred BETWEEN ? AND ? ORDER BY occurred',
                              (site, now-WINDOW, now)).fetchall()
            alerts = db.execute("SELECT lane, MAX(occurred) AS occurred FROM events WHERE site=? AND kind='pothole' "
                                'AND occurred BETWEEN ? AND ? AND lane IS NOT NULL GROUP BY lane',
                                (site, now-WINDOW, now)).fetchall()
        return [tuple(row) for row in rows], {row['lane']: row['occurred'] for row in alerts}

    def passage(self, site, item):
        with closing(self.connect()) as db, db:
            db.execute('INSERT INTO passages VALUES (?,?,?,?,?)', (site,*item))

    def event(self, source, site, event):
        path = event.get('evidence')
        with closing(self.connect()) as db, db:
            db.execute('INSERT INTO events(occurred,source,site,kind,reason,evidence,lane) VALUES(?,?,?,?,?,?,?)',
                       (time.time(), source, site, event['event'], event['reason'],
                        str(Path(path).resolve()) if path else None, event.get('lane')))

    def events(self):
        with closing(self.connect()) as db:
            return [dict(row) for row in db.execute(
                'SELECT id,occurred,source,kind,reason,evidence FROM events ORDER BY id DESC LIMIT 40')]

    def counts(self, site):
        with closing(self.connect()) as db:
            rows = db.execute('SELECT lane,COUNT(*) AS passages,SUM(slow AND NOT merged) AS slowed '
                              'FROM passages WHERE site=? AND occurred BETWEEN ? AND ? GROUP BY lane',
                              (site,time.time()-WINDOW,time.time())).fetchall()
        indexed = {row['lane']: dict(row) for row in rows}
        return [indexed.get(lane, dict(lane=lane, passages=0, slowed=0)) for lane in (0,1)]

import React, { useEffect, useState } from 'react';
import Alert from '../components/Alert';
import Button from '../components/Button';
import Card from '../components/Card';
import { API_BASE_URL } from '../api/config';
import { violationsService } from '../api/services';
import './Violations.css';

const Violations = () => {
  const [violations, setViolations] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchViolations = async () => {
    try {
      setError(null);
      const response = await violationsService.getViolations();
      setViolations(Array.isArray(response) ? response : []);
    } catch (err) {
      setError('Cannot load violation evidence from backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchViolations();
    const interval = setInterval(fetchViolations, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="violations page-shell">
      <div className="page-header">
        <div>
          <h1>Violations</h1>
          <p>Repeat-person denials and captured evidence.</p>
        </div>
        <Button variant="secondary" onClick={fetchViolations}>Refresh</Button>
      </div>

      {error && <Alert type="error" message={error} onClose={() => setError(null)} />}

      <Card>
        {loading ? (
          <div className="empty-state">Loading evidence...</div>
        ) : violations.length === 0 ? (
          <div className="empty-state">No violation evidence captured yet.</div>
        ) : (
          <div className="violation-list">
            {violations.map((group) => (
              <section key={group.groupId} className="violation-group">
                <div className="violation-heading">
                  <div>
                    <h3>{group.userName}</h3>
                    <span>{new Date(group.timestamp).toLocaleString()}</span>
                  </div>
                  <span className="evidence-count">{group.images.length} images</span>
                </div>
                <div className="evidence-grid">
                  {group.images.map((image) => (
                    <button
                      key={image.id}
                      className="evidence-tile"
                      onClick={() => setSelected(`${API_BASE_URL}${image.imageUrl}`)}
                    >
                      <img src={`${API_BASE_URL}${image.imageUrl}`} alt="Violation evidence" />
                    </button>
                  ))}
                </div>
              </section>
            ))}
          </div>
        )}
      </Card>

      {selected && (
        <div className="image-viewer" onClick={() => setSelected(null)}>
          <button className="viewer-close" onClick={() => setSelected(null)}>Close</button>
          <img src={selected} alt="Selected evidence" />
        </div>
      )}
    </div>
  );
};

export default Violations;

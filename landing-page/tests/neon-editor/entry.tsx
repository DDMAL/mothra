import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import NeonBatchEditor from '../../src/components/workflow/NeonBatchEditor';
import type { MeiFile, Project } from '../../src/types';
import '../../src/index.css';

interface Fixture {
  files: MeiFile[];
  initialFileId?: string;
}

export function IntegrationFixture({ files, initialFileId }: Fixture) {
  const [events, setEvents] = useState<string[]>([]);
  const [currentFiles, setCurrentFiles] = useState(files);
  const record = (event: string) => setEvents((previous) => [...previous, event]);
  const project: Project = {
    id: 215,
    name: 'Local embedded Neon integration',
    user: 'cypress',
    images: [], models: [], annotations: [], meiFiles: currentFiles,
    stepsUnlocked: 0,
    usedImageIds: [], usedModelNames: [], usedAnnotationNames: [],
    textAlignments: [], stafflines: [], icXmlFiles: [],
  };
  return <>
    <NeonBatchEditor
      project={project}
      meiFiles={currentFiles}
      initialFileId={initialFileId}
      onBack={() => record('back')}
      onFinish={() => record('finish')}
      onFileCorrected={(id) => {
        // AppRouter updates the file array after correction, which causes
        // the wrapper to prepare fresh sessions. Keep that lifecycle here.
        setCurrentFiles((previous) => previous.map((file) =>
          file.id === id ? { ...file, corrected: true } : file));
        record(`corrected:${id}`);
      }}
    />
    <output id="test-events" hidden>{JSON.stringify(events)}</output>
  </>;
}

// Cypress supplies initial props separately from the mutable content API.
// Keep xmlContent stale, reproducing cached project metadata in the real app.
fetch('/__test__/files').then(async (response) => {
  if (!response.ok) throw new Error('Missing local integration fixture');
  const fixture: Fixture = await response.json();
  createRoot(document.getElementById('root')!).render(<IntegrationFixture {...fixture} />);
});

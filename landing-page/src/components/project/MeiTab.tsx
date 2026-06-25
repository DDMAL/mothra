import { useState } from "react";
import type { Project, MeiFile } from "../../types";
import { authHeaders } from "../../hooks/useAuth";
import { useAssetSection, ITEMS_PER_PAGE } from "../../hooks/useAssetSection";
import { downloadBlob } from "../../utils/download";
import ContextMenu from "../shared/ContextMenu";
import AssetGrid from "../shared/AssetGrid";
import MeiViewerModal from "./MeiViewerModal";
import Modal from "../shared/Modal";

interface MeiTabProps {
  project: Project;
  section: ReturnType<typeof useAssetSection<MeiFile>>;
  onUpdateProject: (p: Project) => void;
  onDeleteMei: (meiId: string) => Promise<void>;
}

export default function MeiTab({
  project,
  section,
  onUpdateProject,
  onDeleteMei,
}: MeiTabProps) {
  const [meiViewFile, setMeiViewFile] = useState<MeiFile | null>(null);
  const [meiSubTab, setMeiSubTab] = useState<"mei produced" | "mei corrected">(
    "mei produced",
  );

  const switchMeiSubTab = (tab: "mei produced" | "mei corrected") => {
    setMeiSubTab(tab);
    section.clearSelection();
    section.setPage(0);
  };

  const handleExportMei = (file: MeiFile) => {
    downloadBlob(
      new Blob([file.xmlContent ?? ""], { type: "application/xml" }),
      file.name,
    );
  };

  const handleValidate = async (file: MeiFile) => {
    setValidateModal({ file, result: null, loading: true });
    const blob = new Blob([file.xmlContent ?? ""], { type: "application/mei" });
    const form = new FormData();
    form.append("file", blob, file.name);
    const r = await fetch("/api/validate-mei", {
      method: "POST",
      headers: authHeaders(),
      body: form,
    });
    const result = await r.json();
    setValidateModal((prev) =>
      prev ? { ...prev, result, loading: false } : null,
    );
  };

  const meiProduced = project.meiFiles.filter((f) => !f.corrected);
  const meiCorrected = project.meiFiles.filter((f) => !!f.corrected);
  const activeMeiFiles =
    meiSubTab === "mei produced" ? meiProduced : meiCorrected;
  const totalMeiPages = Math.ceil(activeMeiFiles.length / ITEMS_PER_PAGE);
  const pagedMei = activeMeiFiles.slice(
    section.page * ITEMS_PER_PAGE,
    (section.page + 1) * ITEMS_PER_PAGE,
  );
  const [validateModal, setValidateModal] = useState<{
    file: MeiFile;
    result: { valid: boolean; warnings: string[] } | null;
    loading: boolean;
  } | null>(null);

  return (
    <>
      <div className="mt-6" onClick={() => section.clearSelection()}>
        <div className="flex gap-2 mb-4">
          {(["mei produced", "mei corrected"] as const).map((sub) => (
            <button
              key={sub}
              onClick={(e) => {
                e.stopPropagation();
                switchMeiSubTab(sub);
              }}
              className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-colors cursor-pointer
                        ${meiSubTab === sub ? "bg-white text-[#4AADAA]" : "text-white/60 hover:text-white/90"}`}
            >
              {sub}
            </button>
          ))}
        </div>
        {activeMeiFiles.length === 0 ? (
          <p className="text-white/70 text-sm">
            {meiSubTab === "mei produced"
              ? "no mei files yet"
              : "no corrected mei files yet"}
          </p>
        ) : (
          <AssetGrid
            pagedItems={pagedMei}
            pageOffset={section.page * ITEMS_PER_PAGE}
            section={section}
            usedNames={[]}
            totalPages={totalMeiPages}
            renderThumbnail={() => (
              <svg
                width="56"
                height="64"
                viewBox="0 0 56 64"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M4 0H36L56 20V60C56 62.2 54.2 64 52 64H4C1.8 64 0 62.2 0 60V4C0 1.8 1.8 0 4 0Z"
                  fill="white"
                  fillOpacity="0.25"
                />
                <path
                  d="M36 0L56 20H40C37.8 20 36 18.2 36 16V0Z"
                  fill="white"
                  fillOpacity="0.45"
                />
                <text
                  x="28"
                  y="46"
                  textAnchor="middle"
                  fill="white"
                  fontSize="14"
                  fontWeight="bold"
                  fontFamily="monospace"
                >
                  MEI
                </text>
              </svg>
            )}
          />
        )}
      </div>

      {section.menu &&
        (() => {
          const file = project.meiFiles.find((f) => f.id === section.menu!.id);
          const newCorrected = file ? !file.corrected : false;
          return (
            <ContextMenu
              x={section.menu.x}
              y={section.menu.y}
              onClose={() => section.setMenu(null)}
              items={[
                {
                  label: "View",
                  onClick: () => {
                    setMeiViewFile(file ?? null);
                    section.setMenu(null);
                  },
                },
                {
                  label: "Export",
                  onClick: () => {
                    const f = project.meiFiles.find(
                      (f) => f.id === section.menu!.id,
                    )!;
                    handleExportMei(f);
                    section.setMenu(null);
                  },
                },
                {
                  label: "Validate",
                  onClick: () => {
                    const f = project.meiFiles.find(
                      (f) => f.id === section.menu!.id,
                    )!;
                    handleValidate(f);
                    section.setMenu(null);
                  },
                },
                {
                  label: "Delete",
                  onClick: async () => {
                    const id = section.menu!.id;
                    section.setMenu(null);
                    await onDeleteMei(id);
                    onUpdateProject({
                      ...project,
                      meiFiles: project.meiFiles.filter((f) => f.id !== id),
                    });
                  },
                },
                ...(file
                  ? [
                      {
                        label: newCorrected
                          ? "Mark as Corrected"
                          : "Mark as Uncorrected",
                        onClick: () => {
                          fetch(`/api/projects/${project.id}/mei/${file.id}`, {
                            method: "PATCH",
                            headers: {
                              ...authHeaders(),
                              "Content-Type": "application/json",
                            },
                            body: JSON.stringify({ corrected: newCorrected }),
                          });
                          onUpdateProject({
                            ...project,
                            meiFiles: project.meiFiles.map((f) =>
                              f.id === file.id
                                ? { ...f, corrected: newCorrected }
                                : f,
                            ),
                          });
                          section.setMenu(null);
                        },
                      },
                    ]
                  : []),
              ]}
            />
          );
        })()}

      {meiViewFile && (
        <MeiViewerModal
          file={meiViewFile}
          project={project}
          onClose={() => setMeiViewFile(null)}
        />
      )}
      {validateModal && (
        <Modal onClose={() => setValidateModal(null)}>
          <h2 className="text-xl text-[#1D3335] text-center">
            {validateModal.loading
              ? "validating..."
              : validateModal.result?.valid
                ? "valid MEI"
                : "MEI has warnings"}
          </h2>
          {!validateModal.loading &&
            validateModal.result &&
            (validateModal.result.valid ? (
              <p className="text-[#1E6B70] text-sm text-center">
                no issues found
              </p>
            ) : (
              <ul className="text-sm text-[#1D3335] flex flex-col gap-1 max-h-64 overflow-y-auto">
                {validateModal.result.warnings.map((w, i) => (
                  <li
                    key={i}
                    className="bg-white/40 rounded-lg px-3 py-1.5 font-mono text-xs"
                  >
                    {w}
                  </li>
                ))}
              </ul>
            ))}
        </Modal>
      )}
    </>
  );
}

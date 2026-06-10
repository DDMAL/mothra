import { useEffect } from "react";

interface DocsQuickStartProps {
    scrollTarget: string | null;
    onNavigateWalkthrough: () => void;
}

export default function DocsQuickStart({ scrollTarget, onNavigateWalkthrough }: DocsQuickStartProps) {
    useEffect(() => {
        if (!scrollTarget) return;
        document.getElementById(`qs-${scrollTarget}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, [scrollTarget]);

    return (
        <div className="animate-fade-in px-10 pt-10 pb-16 text-[#1D3335]">
            <h1 className="text-3xl font-bold italic text-white mb-10">quick start</h1>

            <section id="qs-logging-in" className="mb-10">
                <h2 className="text-2xl font-bold italic text-white mb-4">logging in</h2>
                <p className="leading-relaxed mb-4">
                Users will need to log in in order to access Mothra's features. To log in or register,
                press either the "get started" or "log in" buttons to be brought to the corresponding pages.
                </p>
                <p className="leading-relaxed mb-4">
                If you do not already have an account, you will need to register for one. This will require your:
                </p>
                <ul className="list-disc list-inside mb-4 space-y-1 pl-4">
                <li>Username</li>
                <li>Password</li>
                <li>Email address</li>
                <li>First and last name</li>
                </ul>
                <p className="leading-relaxed">
                After successfully logging in, you are brought to the "my projects" menu, which acts as
                your overall dashboard. This page can also be accessed by clicking the "my projects" button
                at the navbar, which will only appear once you are logged in.
                </p>
            </section>

            <section id="qs-creating-a-project" className="mb-10">
                <h2 className="text-2xl font-bold italic text-white mb-4">creating a project</h2>
                <p className="leading-relaxed mb-4">
                In Mothra, all data is concentrated in <em>projects</em>, which contain <em>images</em>{" "}
                and <em>models</em>, as well as <em>annotations</em> and <em>mei files</em>. You need to
                have at least one project in your dashboard in order to explore a Mothra workflow.
                </p>
                <p className="leading-relaxed mb-4">
                To create a project, simply press the "+ new project" button, located next to the{" "}
                <em>My Projects</em> header. A pop-up will appear asking for the name of the project,
                which can always be changed later.
                </p>
                <div className="w-full h-64 bg-[#1E6B70] rounded-lg mb-4" />
                <p className="leading-relaxed mb-4">
                To rename a project from the project dashboard, you can press the pencil icon that appears
                when hovering over the project in the main dashboard. Similarly, to delete a project, you
                can press the trashcan icon. Deleting a project will bring it to a "trash" folder, accessed
                via the dashboard; you can restore a project for up to 30 days until it is permanently deleted.
                </p>
                <p className="leading-relaxed">
                Now that a project has successfully been created, you can click on the project name to be
                brought to the individual project detail page. This contains information such as user-uploaded
                images and models, and the projects' progress through the Mothra workflow. As you progress,
                you will be able to go back to previous steps.
                </p>
            </section>

            <section id="qs-uploading-images" className="mb-10">
                <h2 className="text-2xl font-bold italic text-white mb-4">uploading images</h2>
                <p className="leading-relaxed mb-4">
                In order to upload an image, navigate to the "images" tab and press the button "+ new image"
                to add a new image to the project. You will be prompted to either drag and drop a supported
                file, or select one from your directory.
                </p>
                <p className="leading-relaxed mb-4">
                All images will upload with names being their names within your computer. PDFs will
                automatically be parsed into separate pages, with page number indicated in their names. Of
                course, you have the option to rename or delete certain images, either by clicking the three
                dots next to the image name or clicking an image (deletion only).
                </p>
                <p className="leading-relaxed">
                Additionally, clicking an image gives you the option to "use" it in the Mothra workflow.
                Opting to use an image will place it in the menu underneath the "continue" button.
                </p>
            </section>

            <section id="qs-uploading-models" className="mb-10">
                <h2 className="text-2xl font-bold italic text-white mb-4">uploading models</h2>
                <p className="leading-relaxed mb-4">
                Some models have already been uploaded, as part of the DDMAL lab's work; you are free to
                select one of these "default" models, or upload one of your own.
                </p>
                <p className="leading-relaxed mb-4">
                Similar to images, in order to upload a model, navigate to the "models" tab and press the
                button "+ new model" to add a new model to the project. You will be prompted to either drag
                and drop a supported file, or select one from your directory.
                </p>
                <p className="leading-relaxed mb-4">
                All models will upload with names being their names within your computer. Of course, you have
                the option to rename or delete certain models, either by clicking the three dots next to the
                model name or clicking a model (deletion only).
                </p>
                <p className="leading-relaxed">
                Additionally, clicking a model gives you the option to "use" it in the Mothra workflow.
                Opting to use a model will place it in the menu underneath the "continue" button.
                </p>
            </section>

            <section id="qs-what-now" className="mb-10">
                <h2 className="text-2xl font-bold italic text-white mb-4">what now?</h2>
                <p className="leading-relaxed mb-4">
                Once you have selected a model and at least one image, you're all set to travel through the
                Mothra workflow by pressing "continue", thus concluding your "quick start" to Mothra!
                </p>
                <p className="leading-relaxed mb-6">
                Please refer to the walkthrough page, in order to read detailed information on each step.
                </p>
                <div className="w-full h-64 bg-[#1E6B70] rounded-lg mb-8" />
            </section>

            <div className="flex justify-end">
                <button
                onClick={onNavigateWalkthrough}
                className="text-sm hover:opacity-70 transition-opacity"
                >
                continue to walkthrough →
                </button>
            </div>
        </div>
    );
}
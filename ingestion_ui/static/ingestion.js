let token = document.querySelector('meta[name="ingestion-token"]').content;
const dateFormat = new Intl.DateTimeFormat(undefined, {
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
});

async function api(path, method = "GET", data) {
    const headers = method === "GET"
        ? {}
        : {"Content-Type": "application/json", "X-Ingestion-Token": token};
    const response = await fetch(path, {
        method,
        headers,
        body: data === undefined ? undefined : JSON.stringify(data),
        signal: AbortSignal.timeout(8000),
    });
    if (!response.ok) {
        const result = await response.json().catch(() => ({}));
        throw new Error(result.error || "The console could not complete the request.");
    }
    return response.json();
}

Vue.createApp({
    compilerOptions: {delimiters: ["[[", "]]"]},
    data() {
        return {
            mode: "once", path: "", jobs: [], watchers: [], phase: "loading", loaded: false,
            refreshing: false, submitting: false, pending: new Set(), timer: null,
            formNotice: "", formError: "", connectionError: "", watchError: "", jobError: "",
        };
    },
    computed: {
        watching() {
            return this.mode === "watch";
        },
        connectionLabel() {
            return {ready: "Connected", offline: "Connection lost", loading: "Connecting"}[this.phase];
        },
        submitLabel() {
            if (this.submitting) return this.watching ? "Adding watch…" : "Queuing…";
            return this.watching ? "Add watch" : "Queue embedding job";
        },
    },
    watch: {
        mode() {
            this.formNotice = "";
            this.formError = "";
        },
    },
    mounted() {
        this.refresh();
        this.timer = setInterval(() => this.refresh(), 1500);
    },
    beforeUnmount() {
        clearInterval(this.timer);
    },
    methods: {
        async refresh() {
            if (this.refreshing) return;
            this.refreshing = true;
            try {
                const {jobs, watchers, mutation_token} = await api("/api/dashboard");
                token = mutation_token;
                this.jobs = jobs;
                this.watchers = watchers;
                this.loaded = true;
                this.phase = "ready";
            } catch {
                this.phase = "offline";
                this.connectionError = "The local ingestion service is unreachable.";
            } finally {
                this.refreshing = false;
            }
        },
        clearFormFeedback() {
            this.formNotice = "";
            this.formError = "";
        },
        async submitSource() {
            if (this.submitting) return;
            this.clearFormFeedback();
            const path = this.path.trim();
            if (!path) {
                this.formError = "Enter an absolute path.";
                this.$refs.pathInput.focus();
                return;
            }
            this.submitting = true;
            try {
                await api(this.watching ? "/api/watchers" : "/api/jobs", "POST", {path});
                this.formNotice = this.watching ? "Watch added." : "Embedding job queued.";
                this.path = "";
                await this.refresh();
            } catch (error) {
                this.formError = error.message;
                this.$refs.pathInput.focus();
            } finally {
                this.submitting = false;
            }
        },
        async toggleWatch(watch) {
            this.watchError = "";
            this.pending.add(watch.id);
            try {
                const {enabled} = await api(`/api/watchers/${watch.id}`, "PATCH", {enabled: !watch.enabled});
                watch.enabled = enabled;
                await this.refresh();
            } catch (error) {
                this.watchError = error.message;
            } finally {
                this.pending.delete(watch.id);
            }
        },
        async retry(job) {
            this.jobError = "";
            this.pending.add(job.id);
            try {
                await api(`/api/jobs/${job.id}/retry`, "POST", {});
                await this.refresh();
            } catch (error) {
                this.jobError = error.message;
            } finally {
                this.pending.delete(job.id);
            }
        },
        statusLabel(status) {
            return {queued: "Queued", running: "Running", completed: "Completed", failed: "Failed"}[status];
        },
        formatDate(value) {
            return dateFormat.format(new Date(value));
        },
        describeResult(result) {
            if (!result.scanned_files) return "No supported content found";
            if (!result.indexed_files && result.skipped_unchanged) return `${result.skipped_unchanged} unchanged`;
            const files = result.indexed_files === 1 ? "file" : "files";
            const chunks = result.total_chunks_upserted === 1 ? "chunk" : "chunks";
            return `${result.indexed_files} ${files} · ${result.total_chunks_upserted} ${chunks}`;
        },
    },
}).mount("#console");

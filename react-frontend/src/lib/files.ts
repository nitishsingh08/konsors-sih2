/* =====================================================================
   Saving and copying files. Uses the host's downloads API when the page
   is running inside a sandbox that offers one, and falls back to a blob
   download, then to the clipboard.
   ===================================================================== */
import { useToast } from '../state/ToastProvider';

interface HostDownloads {
  save: (a: { filename: string; data: string }) => Promise<void>;
}

interface HostWindow {
  claude?: { use?: (name: string) => Promise<unknown> };
}

let hostDownloads: HostDownloads | null = null;
let probed = false;

async function hostDls(): Promise<HostDownloads | null> {
  if (probed) return hostDownloads;
  probed = true;
  try {
    const w = window as unknown as HostWindow;
    if (w.claude && w.claude.use) hostDownloads = (await w.claude.use('downloads')) as HostDownloads;
  } catch {
    hostDownloads = null;
  }
  return hostDownloads;
}

export function useFiles() {
  const toast = useToast();

  const copyText = async (txt: string, okMsg = 'Copied') => {
    try {
      await navigator.clipboard.writeText(txt);
      toast(okMsg);
      return;
    } catch {
      /* fall through to the legacy path */
    }
    try {
      const ta = document.createElement('textarea');
      ta.value = txt;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
      toast(okMsg);
    } catch {
      toast('Could not copy in this view');
    }
  };

  const saveFile = async (filename: string, data: string, label = 'File') => {
    const dls = await hostDls();
    const inHost = !!(window as unknown as HostWindow).claude;
    if (dls) {
      try {
        await dls.save({ filename, data });
        toast('Saved ' + filename);
      } catch (e) {
        if (!e || (e as { code?: string }).code !== 'declined') toast('Could not save the file');
      }
      return;
    }
    if (!inHost) {
      try {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([data], { type: 'text/plain;charset=utf-8' }));
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 1500);
        toast('Saved ' + filename);
        return;
      } catch {
        await copyText(data, label + ' copied to clipboard');
        return;
      }
    }
    await copyText(data, label + ' copied to clipboard (saving files is not available in this view)');
  };

  return { copyText, saveFile };
}

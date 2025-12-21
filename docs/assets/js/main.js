(() => {
  const navToggle = document.querySelector('[data-nav-toggle]');
  const nav = document.querySelector('[data-nav]');

  if (navToggle && nav) {
    navToggle.addEventListener('click', () => {
      const isOpen = nav.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', String(isOpen));
    });

    nav.addEventListener('click', (e) => {
      const target = e.target;
      if (target && target.matches('a[href^="#"]')) {
        nav.classList.remove('open');
        navToggle.setAttribute('aria-expanded', 'false');
      }
    });
  }

  const tabs = Array.from(document.querySelectorAll('[data-tab]'));
  const panels = Array.from(document.querySelectorAll('[data-panel]'));

  const resetVideo = (video) => {
    if (!video) return;
    try {
      video.pause();
      video.currentTime = 0;
    } catch {
      // ignore
    }
  };

  const setActive = (key, pushHash) => {
    const activeTab = tabs.find((t) => t.dataset.tab === key) || tabs[0];
    const activeKey = activeTab?.dataset.tab;

    tabs.forEach((t) => t.setAttribute('aria-selected', String(t.dataset.tab === activeKey)));
    panels.forEach((p) => p.classList.toggle('active', p.dataset.panel === activeKey));

    // Ensure hidden panels don't keep playing videos and all previews reset to the start.
    panels
      .filter((p) => p.dataset.panel !== activeKey)
      .forEach((p) => p.querySelectorAll('video').forEach(resetVideo));

    // Intentionally do not update window.location.hash here.
    // We want hashes to stay available for section navigation (e.g., #results).
    void pushHash;
  };

  if (tabs.length && panels.length) {
    tabs.forEach((t) => {
      t.addEventListener('click', () => setActive(t.dataset.tab, false));
    });

    const fromHash = window.location.hash?.replace('#', '');
    if (fromHash && tabs.some((t) => t.dataset.tab === fromHash)) {
      setActive(fromHash, false);
    } else {
      setActive(tabs[0].dataset.tab, false);
    }
  }

  // Improve table UX: ensure any raw <table> under dataset panels gets wrapped
  // in a horizontally scrollable container.
  const wrapTables = () => {
    const tables = Array.from(document.querySelectorAll('[data-panel] table'));
    tables.forEach((table) => {
      const parent = table.parentElement;
      if (!parent) return;
      if (parent.classList.contains('table-wrap')) return;

      const wrap = document.createElement('div');
      wrap.className = 'table-wrap';
      parent.insertBefore(wrap, table);
      wrap.appendChild(table);
    });
  };

  wrapTables();

  // Theme toggle (GitHub Pages safe): defaults to system via CSS
  // (prefers-color-scheme). Users can override and persist their choice.
  const themeToggle = document.querySelector('[data-theme-toggle]');
  const root = document.documentElement;
  const storageKey = 'g3splat-theme';
  const allowed = new Set(['system', 'light', 'dark']);

  const getStoredTheme = () => {
    try {
      const value = localStorage.getItem(storageKey) || 'system';
      return allowed.has(value) ? value : 'system';
    } catch {
      return 'system';
    }
  };

  const storeTheme = (value) => {
    try {
      localStorage.setItem(storageKey, value);
    } catch {
      // ignore (private mode, disabled storage)
    }
  };

  const applyTheme = (value) => {
    if (value === 'system') {
      root.removeAttribute('data-theme');
    } else {
      root.setAttribute('data-theme', value);
    }

    if (themeToggle) {
      const label = value === 'system' ? 'System' : value[0].toUpperCase() + value.slice(1);
      themeToggle.textContent = `Theme: ${label}`;
      themeToggle.setAttribute('aria-label', `Theme: ${label}`);
    }
  };

  if (themeToggle) {
    const initial = getStoredTheme();
    applyTheme(initial);

    themeToggle.addEventListener('click', () => {
      const current = getStoredTheme();
      const next = current === 'system' ? 'light' : current === 'light' ? 'dark' : 'system';
      storeTheme(next);
      applyTheme(next);
    });
  }
})();

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

  // Only initialize actual tablists.
  // Panels also carry `data-tabset` (to associate them to a tabset key), so
  // selecting all `[data-tabset]` would incorrectly bind nested tabs and cause
  // panels to disappear (e.g., Geometry → Depth/Mesh).
  const getTabsets = () => Array.from(document.querySelectorAll('[role="tablist"][data-tabset]'));

  const resetVideo = (video) => {
    if (!video) return;
    try {
      video.pause();
      video.currentTime = 0;
    } catch {
      // ignore
    }
  };

  const playAutoplayVideos = (rootEl) => {
    if (!rootEl) return;
    const videos = Array.from(rootEl.querySelectorAll('video[autoplay]'));
    videos.forEach((video) => {
      try {
        // Ensure we satisfy common autoplay requirements.
        // (User gesture from the tab click + muted is usually enough.)
        video.muted = true;
        video.playsInline = true;
        const p = video.play();
        if (p && typeof p.catch === 'function') p.catch(() => {});
      } catch {
        // ignore
      }
    });
  };

  const initTabset = (tabsetEl) => {
    const tabsetKey = tabsetEl.dataset.tabset;
    if (!tabsetKey) return;

    const tabs = Array.from(tabsetEl.querySelectorAll('[data-tab]'));
    const panels = Array.from(document.querySelectorAll(`[data-panel][data-tabset="${tabsetKey}"]`));
    if (!tabs.length || !panels.length) return;

    const setActive = (key) => {
      const activeTab = tabs.find((t) => t.dataset.tab === key) || tabs.find((t) => t.getAttribute('aria-selected') === 'true') || tabs[0];
      const activeKey = activeTab?.dataset.tab;

      tabs.forEach((t) => {
        const isActive = t.dataset.tab === activeKey;
        t.setAttribute('aria-selected', String(isActive));
        // Improve keyboard navigation: only the active tab is focusable.
        t.tabIndex = isActive ? 0 : -1;
      });
      panels.forEach((p) => p.classList.toggle('active', p.dataset.panel === activeKey));

      // Ensure hidden panels don't keep playing videos and all previews reset to the start.
      panels
        .filter((p) => p.dataset.panel !== activeKey)
        .forEach((p) => p.querySelectorAll('video').forEach(resetVideo));

      // If the newly-active panel contains videos, start them.
      // Autoplay does not reliably trigger when videos become visible later.
      const activePanel = panels.find((p) => p.dataset.panel === activeKey);
      playAutoplayVideos(activePanel);
    };

    tabs.forEach((t) => {
      t.addEventListener('click', () => setActive(t.dataset.tab));
    });

    const fromHash = window.location.hash?.replace('#', '');
    if (fromHash && tabs.some((t) => t.dataset.tab === fromHash)) {
      setActive(fromHash);
    } else {
      const initiallySelected = tabs.find((t) => t.getAttribute('aria-selected') === 'true') || tabs[0];
      setActive(initiallySelected.dataset.tab);
    }
  };

  const tabsets = getTabsets();
  tabsets.forEach(initTabset);

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

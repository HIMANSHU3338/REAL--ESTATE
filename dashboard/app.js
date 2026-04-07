/* ═══════════════════════════════════════════════════════════
   Real Estate RL Dashboard — Application Logic
   Indian Market Edition 🇮🇳 — All prices in ₹
   ═══════════════════════════════════════════════════════════ */

// ─── Chart instances ───
let chartNetWorth = null;
let chartRegime = null;
let chartActions = null;
let chartReturnsBar = null;
let chartInterest = null;
let chartDemand = null;
let chartCashflow = null;

// ─── INR multiplier (environment uses base values, display in ₹) ───
const INR_MULTIPLIER = 1; // Values are already in INR from the env

// ─── Chart.js Global Config ───
Chart.defaults.color = '#a5b4fc';
Chart.defaults.borderColor = 'rgba(99, 102, 241, 0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(5, 8, 16, 0.95)';
Chart.defaults.plugins.tooltip.borderColor = 'rgba(99, 102, 241, 0.2)';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.cornerRadius = 10;
Chart.defaults.plugins.tooltip.padding = 14;
Chart.defaults.plugins.tooltip.titleFont = { weight: '700', size: 12 };
Chart.defaults.elements.point.radius = 0;
Chart.defaults.elements.point.hoverRadius = 6;
Chart.defaults.elements.point.hoverBorderWidth = 2;

// ─── Color Palette ───
const COLORS = {
    green: '#34d399',
    greenAlpha: 'rgba(52, 211, 153, 0.15)',
    red: '#f87171',
    redAlpha: 'rgba(248, 113, 113, 0.12)',
    blue: '#818cf8',
    blueAlpha: 'rgba(129, 140, 248, 0.12)',
    amber: '#fbbf24',
    amberAlpha: 'rgba(251, 191, 36, 0.12)',
    purple: '#a78bfa',
    purpleAlpha: 'rgba(167, 139, 250, 0.12)',
    pink: '#f472b6',
    cyan: '#22d3ee',
    gray: '#6366a0',
    grayAlpha: 'rgba(99, 102, 160, 0.12)',
};

const AGENT_COLORS = {
    'Random': { main: COLORS.gray, alpha: COLORS.grayAlpha },
    'BuyAndHold': { main: COLORS.blue, alpha: COLORS.blueAlpha },
    'RuleBased': { main: COLORS.amber, alpha: COLORS.amberAlpha },
    'PPO_Trained': { main: COLORS.green, alpha: COLORS.greenAlpha },
};

const REGIME_COLORS = {
    'BOOM': COLORS.green,
    'STABLE': COLORS.amber,
    'RECESSION': COLORS.red,
};

// ─── Data Store ───
let evaluationData = null;
let demoData = null;

// ═══════════════════════════════════════════════════════════
// INDIAN RUPEE FORMATTING
// ═══════════════════════════════════════════════════════════

function formatINR(value) {
    const absVal = Math.abs(value);
    const sign = value < 0 ? '-' : '';

    if (absVal >= 10000000) {
        // Crores
        return sign + '₹' + (absVal / 10000000).toFixed(2) + ' Cr';
    } else if (absVal >= 100000) {
        // Lakhs
        return sign + '₹' + (absVal / 100000).toFixed(2) + ' L';
    } else if (absVal >= 1000) {
        // Thousands
        return sign + '₹' + (absVal / 1000).toFixed(1) + 'K';
    } else {
        return sign + '₹' + Math.round(absVal).toLocaleString('en-IN');
    }
}

function formatINRFull(value) {
    return '₹' + Math.round(value).toLocaleString('en-IN');
}

function formatINRAxis(value) {
    const absVal = Math.abs(value);
    if (absVal >= 10000000) return '₹' + (absVal / 10000000).toFixed(1) + 'Cr';
    if (absVal >= 100000) return '₹' + (absVal / 100000).toFixed(0) + 'L';
    if (absVal >= 1000) return '₹' + (absVal / 1000).toFixed(0) + 'K';
    return '₹' + Math.round(absVal);
}

// ═══════════════════════════════════════════════════════════
// DATA LOADING
// ═══════════════════════════════════════════════════════════

async function loadAllData() {
    const btn = document.getElementById('btn-load-data');
    btn.innerHTML = '<span class="btn-icon">⏳</span> Loading...';

    try {
        const evalResp = await fetch('../results/evaluation_results.json');
        if (evalResp.ok) {
            evaluationData = await evalResp.json();
            console.log('✅ Loaded evaluation data', evaluationData);
        }
    } catch (e) {
        console.warn('Could not load evaluation_results.json:', e.message);
    }

    try {
        const demoResp = await fetch('../results/demo_episode.json');
        if (demoResp.ok) {
            demoData = await demoResp.json();
            console.log('✅ Loaded demo data', demoData);
        }
    } catch (e) {
        console.warn('Could not load demo_episode.json:', e.message);
    }

    if (!evaluationData && !demoData) {
        console.log('📊 Using built-in Indian market sample data');
        useSampleData();
    }

    renderAll();
    btn.innerHTML = '<span class="btn-icon">✅</span> Loaded!';
    setTimeout(() => {
        btn.innerHTML = '<span class="btn-icon">📂</span> Load Data';
    }, 2500);
}

function useSampleData() {
    // Indian market pricing: values in ₹
    evaluationData = {
        "Random": {
            "agent": "Random", "n_episodes": 50,
            "avg_return_pct": -79.73, "std_return_pct": 6.8,
            "avg_sharpe": -1.412, "avg_max_drawdown_pct": 80.69,
            "avg_reward": 14.873, "avg_final_net_worth": 4053320,
            "win_rate_pct": 0.0, "best_return_pct": -53.8, "worst_return_pct": -91.42,
            "action_distribution": { "Hold": 20.0, "Raise Rent": 20.2, "Sell": 20.1, "Lower Rent": 19.6, "Buy": 20.0 }
        },
        "BuyAndHold": {
            "agent": "BuyAndHold", "n_episodes": 50,
            "avg_return_pct": 143.76, "std_return_pct": 52.76,
            "avg_sharpe": 1.808, "avg_max_drawdown_pct": 8.6,
            "avg_reward": 1.63, "avg_final_net_worth": 48751120,
            "win_rate_pct": 100.0, "best_return_pct": 257.73, "worst_return_pct": 15.87,
            "action_distribution": { "Buy": 5.4, "Hold": 94.6 }
        },
        "RuleBased": {
            "agent": "RuleBased", "n_episodes": 50,
            "avg_return_pct": -9.24, "std_return_pct": 27.69,
            "avg_sharpe": -0.08, "avg_max_drawdown_pct": 44.38,
            "avg_reward": 0.3368, "avg_final_net_worth": 18152760,
            "win_rate_pct": 36.0, "best_return_pct": 64.55, "worst_return_pct": -51.23,
            "action_distribution": { "Buy": 6.2, "Hold": 80.7, "Raise Rent": 8.9, "Sell": 3.7, "Lower Rent": 0.4 }
        }
    };

    demoData = generateSyntheticDemo();
}

function generateSyntheticDemo() {
    const months = 120;
    const initialCapital = 20000000; // ₹2 Crore
    const netWorthHistory = [initialCapital];
    const regimeHistory = ['STABLE'];
    const log = [];

    let nw = initialCapital;
    let cash = initialCapital;
    let regime = 'STABLE';
    let rate = 0.065; // RBI repo-like
    let demand = 0.5;
    let properties = 0;

    for (let m = 1; m <= months; m++) {
        if (Math.random() < 0.08) {
            const r = Math.random();
            if (r < 0.35) regime = 'BOOM';
            else if (r < 0.7) regime = 'STABLE';
            else regime = 'RECESSION';
        }
        regimeHistory.push(regime);

        const drift = regime === 'BOOM' ? 0.006 : regime === 'RECESSION' ? -0.003 : 0.002;
        rate = Math.max(0.04, Math.min(0.12, rate + (Math.random() - 0.5) * 0.003));
        demand = Math.max(0, Math.min(1, demand + (Math.random() - 0.5) * 0.1));

        const noise = (Math.random() - 0.5) * 0.015;
        nw = nw * (1 + drift + noise);
        if (m <= 3 && properties < 3) { nw -= 2000000; cash -= 2000000; properties++; }
        cash += (Math.random() - 0.3) * 30000;
        netWorthHistory.push(Math.round(nw));

        const actions = [];
        for (let s = 0; s < 5; s++) {
            const acts = ['Hold', 'Buy', 'Sell', 'Raise Rent', 'Lower Rent'];
            actions.push({
                slot: s,
                action: acts[Math.floor(Math.random() * 5)],
                success: Math.random() > 0.3,
                message: ''
            });
        }

        log.push({
            month: m,
            regime: regime,
            actions: actions,
            cash_after: Math.round(cash),
            net_worth: Math.round(nw),
            market: { regime, interest_rate: +rate.toFixed(4), demand: +demand.toFixed(3), inflation: 0.06 },
        });
    }

    return {
        summary: {
            initial_capital: initialCapital,
            final_net_worth: Math.round(nw),
            total_return_pct: +((nw - initialCapital) / initialCapital * 100).toFixed(2),
            annualized_return_pct: +(((nw / initialCapital) ** (1/10) - 1) * 100).toFixed(2),
            annualized_sharpe: 1.5,
            max_drawdown_pct: 12.8,
            total_properties_bought: 14,
            total_properties_sold: 9,
            total_reward: 3.12,
        },
        net_worth_history: netWorthHistory,
        regime_history: regimeHistory,
        log: log,
    };
}

// ═══════════════════════════════════════════════════════════
// RENDER ALL
// ═══════════════════════════════════════════════════════════

function renderAll() {
    renderKPIs();
    renderNetWorthChart();
    renderRegimeChart();
    renderComparisonTable();
    renderActionsChart();
    renderReturnsBarChart();
    renderMarketCharts();
    renderActionLog();
}

// ═══════════════════════════════════════════════════════════
// KPI RENDERING
// ═══════════════════════════════════════════════════════════

function renderKPIs() {
    if (!demoData && !evaluationData) return;

    const summary = demoData?.summary || {};
    const bestAgent = evaluationData ? getBestAgent() : null;
    const src = bestAgent ? evaluationData[bestAgent] : {};

    // Net Worth
    const finalNW = summary.final_net_worth || src.avg_final_net_worth || 0;
    document.getElementById('kpi-nw-value').textContent = formatINR(finalNW);
    const returnPct = summary.total_return_pct || src.avg_return_pct || 0;
    const nwChange = document.getElementById('kpi-nw-change');
    nwChange.textContent = (returnPct >= 0 ? '+' : '') + returnPct.toFixed(1) + '%';
    nwChange.className = 'kpi-change ' + (returnPct >= 0 ? 'positive' : 'negative');

    // Return
    document.getElementById('kpi-return-value').textContent = (returnPct >= 0 ? '+' : '') + returnPct.toFixed(1) + '%';
    const ann = summary.annualized_return_pct || 0;
    const retAnn = document.getElementById('kpi-return-ann');
    retAnn.textContent = 'Ann. ' + (ann >= 0 ? '+' : '') + ann.toFixed(1) + '%';
    retAnn.className = 'kpi-change ' + (ann >= 0 ? 'positive' : 'negative');

    // Sharpe
    const sharpe = summary.annualized_sharpe || src.avg_sharpe || 0;
    document.getElementById('kpi-sharpe-value').textContent = sharpe.toFixed(3);
    const sq = document.getElementById('kpi-sharpe-quality');
    if (sharpe > 1.5) { sq.textContent = '🔥 Excellent'; sq.className = 'kpi-change positive'; }
    else if (sharpe > 0.5) { sq.textContent = '✅ Good'; sq.className = 'kpi-change neutral'; }
    else { sq.textContent = '⚠️ Poor'; sq.className = 'kpi-change negative'; }

    // Max Drawdown
    const dd = summary.max_drawdown_pct || src.avg_max_drawdown_pct || 0;
    document.getElementById('kpi-dd-value').textContent = dd.toFixed(1) + '%';
    const ddRisk = document.getElementById('kpi-dd-risk');
    if (dd < 15) { ddRisk.textContent = '🛡️ Low Risk'; ddRisk.className = 'kpi-change positive'; }
    else if (dd < 40) { ddRisk.textContent = '⚡ Medium'; ddRisk.className = 'kpi-change neutral'; }
    else { ddRisk.textContent = '🔴 High Risk'; ddRisk.className = 'kpi-change negative'; }

    // Properties
    const bought = summary.total_properties_bought || 0;
    const sold = summary.total_properties_sold || 0;
    document.getElementById('kpi-prop-value').textContent = bought + ' / ' + sold;

    // Reward
    const reward = summary.total_reward || src.avg_reward || 0;
    document.getElementById('kpi-reward-value').textContent = (typeof reward === 'number' ? reward.toFixed(3) : reward);
}

// ═══════════════════════════════════════════════════════════
// NET WORTH CHART
// ═══════════════════════════════════════════════════════════

function renderNetWorthChart() {
    const ctx = document.getElementById('chart-net-worth');
    if (!ctx) return;
    if (chartNetWorth) chartNetWorth.destroy();

    const nwHistory = demoData?.net_worth_history || [];
    if (nwHistory.length === 0) return;

    const labels = nwHistory.map((_, i) => i === 0 ? 'Start' : 'M' + i);
    const initial = nwHistory[0] || 20000000;

    const context = ctx.getContext('2d');
    const gradient = context.createLinearGradient(0, 0, 0, 290);
    const isPositive = nwHistory[nwHistory.length - 1] >= initial;
    if (isPositive) {
        gradient.addColorStop(0, 'rgba(52, 211, 153, 0.35)');
        gradient.addColorStop(0.5, 'rgba(52, 211, 153, 0.08)');
        gradient.addColorStop(1, 'rgba(52, 211, 153, 0.0)');
    } else {
        gradient.addColorStop(0, 'rgba(248, 113, 113, 0.35)');
        gradient.addColorStop(0.5, 'rgba(248, 113, 113, 0.08)');
        gradient.addColorStop(1, 'rgba(248, 113, 113, 0.0)');
    }

    const badge = document.getElementById('nw-trend-badge');
    const totalReturn = ((nwHistory[nwHistory.length-1] - initial) / initial * 100).toFixed(1);
    badge.textContent = (totalReturn >= 0 ? '↑ +' : '↓ ') + totalReturn + '%';
    badge.style.background = isPositive ? 'rgba(52,211,153,0.12)' : 'rgba(248,113,113,0.12)';
    badge.style.color = isPositive ? COLORS.green : COLORS.red;

    chartNetWorth = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Net Worth (₹)',
                    data: nwHistory,
                    borderColor: isPositive ? COLORS.green : COLORS.red,
                    backgroundColor: gradient,
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.4,
                    pointHoverBackgroundColor: isPositive ? COLORS.green : COLORS.red,
                    pointHoverBorderColor: '#fff',
                },
                {
                    label: 'Initial Capital',
                    data: new Array(nwHistory.length).fill(initial),
                    borderColor: 'rgba(163, 167, 214, 0.2)',
                    borderWidth: 1.5,
                    borderDash: [8, 5],
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: ctx => ctx.dataset.label + ': ' + formatINRFull(ctx.parsed.y)
                    }
                }
            },
            scales: {
                x: {
                    ticks: { maxTicksLimit: 12, font: { size: 10 } },
                    grid: { display: false },
                },
                y: {
                    ticks: {
                        callback: v => formatINRAxis(v),
                        font: { size: 10 },
                    },
                }
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════
// REGIME TIMELINE
// ═══════════════════════════════════════════════════════════

function renderRegimeChart() {
    const ctx = document.getElementById('chart-regime');
    if (!ctx) return;
    if (chartRegime) chartRegime.destroy();

    const regimes = demoData?.regime_history || [];
    if (regimes.length === 0) return;

    const labels = regimes.map((_, i) => 'M' + i);
    const regimeValues = regimes.map(r => r === 'BOOM' ? 3 : r === 'STABLE' ? 2 : 1);
    const regimeColorArr = regimes.map(r => REGIME_COLORS[r] || COLORS.gray);

    chartRegime = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Market Regime',
                data: regimeValues,
                backgroundColor: regimeColorArr.map(c => c + '66'),
                borderColor: regimeColorArr.map(c => c + 'CC'),
                borderWidth: 1,
                borderRadius: 3,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: ctx => 'Month ' + ctx[0].dataIndex,
                        label: ctx => regimes[ctx.dataIndex]
                    }
                }
            },
            scales: {
                x: {
                    ticks: { maxTicksLimit: 12, font: { size: 10 } },
                    grid: { display: false },
                },
                y: {
                    min: 0, max: 4,
                    ticks: {
                        stepSize: 1,
                        callback: v => ['', '🔴 Recession', '🟡 Stable', '🟢 Boom', ''][v] || '',
                        font: { size: 10 },
                    }
                }
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════
// COMPARISON TABLE
// ═══════════════════════════════════════════════════════════

function renderComparisonTable() {
    if (!evaluationData) return;

    const tbody = document.getElementById('comparison-tbody');
    tbody.innerHTML = '';

    const bestAgent = getBestAgent();

    for (const [name, data] of Object.entries(evaluationData)) {
        const tr = document.createElement('tr');
        if (name === bestAgent) tr.classList.add('best-row');

        const isBest = name === bestAgent;
        const badgeHTML = isBest ? '<span class="agent-badge badge-winner">🏆 BEST</span>' : '';
        const agentIcon = { Random: '🎲', BuyAndHold: '🏠', RuleBased: '🧠', PPO_Trained: '🤖' };

        tr.innerHTML = `
            <td><span class="agent-name">${agentIcon[name] || '🤖'} ${name} ${badgeHTML}</span></td>
            <td class="${data.avg_return_pct >= 0 ? 'val-positive' : 'val-negative'}">${data.avg_return_pct >= 0 ? '+' : ''}${data.avg_return_pct.toFixed(1)}%</td>
            <td class="${data.avg_sharpe >= 1 ? 'val-positive' : data.avg_sharpe < 0 ? 'val-negative' : 'val-neutral'}">${data.avg_sharpe.toFixed(3)}</td>
            <td class="${data.avg_max_drawdown_pct < 20 ? 'val-positive' : 'val-negative'}">${data.avg_max_drawdown_pct.toFixed(1)}%</td>
            <td class="${data.win_rate_pct > 50 ? 'val-positive' : data.win_rate_pct > 0 ? 'val-neutral' : 'val-negative'}">${data.win_rate_pct.toFixed(0)}%</td>
            <td>${formatINR(data.avg_final_net_worth)}</td>
            <td class="val-positive">+${data.best_return_pct?.toFixed(1) || 0}%</td>
            <td class="val-negative">${data.worst_return_pct?.toFixed(1) || 0}%</td>
        `;
        tbody.appendChild(tr);
    }
}

// ═══════════════════════════════════════════════════════════
// ACTION DISTRIBUTION CHART
// ═══════════════════════════════════════════════════════════

function renderActionsChart() {
    const ctx = document.getElementById('chart-actions');
    if (!ctx || !evaluationData) return;
    if (chartActions) chartActions.destroy();

    const actionNames = ['Hold', 'Buy', 'Sell', 'Raise Rent', 'Lower Rent'];
    const actionColors = [COLORS.gray, COLORS.green, COLORS.red, COLORS.blue, COLORS.amber];
    const agents = Object.keys(evaluationData);

    const datasets = actionNames.map((action, i) => ({
        label: action,
        data: agents.map(a => evaluationData[a].action_distribution?.[action] || 0),
        backgroundColor: actionColors[i] + 'BB',
        borderColor: actionColors[i],
        borderWidth: 1,
        borderRadius: 4,
    }));

    chartActions = new Chart(ctx, {
        type: 'bar',
        data: { labels: agents, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: { label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%' }
                }
            },
            scales: {
                x: { stacked: true, grid: { display: false } },
                y: {
                    stacked: true,
                    max: 100,
                    ticks: { callback: v => v + '%', font: { size: 10 } },
                }
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════
// RETURNS BAR CHART
// ═══════════════════════════════════════════════════════════

function renderReturnsBarChart() {
    const ctx = document.getElementById('chart-returns-bar');
    if (!ctx || !evaluationData) return;
    if (chartReturnsBar) chartReturnsBar.destroy();

    const agents = Object.keys(evaluationData);
    const avgReturns = agents.map(a => evaluationData[a].avg_return_pct);
    const stdReturns = agents.map(a => evaluationData[a].std_return_pct);

    const barColors = agents.map(a => {
        const c = AGENT_COLORS[a] || AGENT_COLORS['Random'];
        return c.main;
    });

    chartReturnsBar = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: agents,
            datasets: [
                {
                    label: 'Avg Return (%)',
                    data: avgReturns,
                    backgroundColor: barColors.map(c => c + 'AA'),
                    borderColor: barColors,
                    borderWidth: 2,
                    borderRadius: 10,
                },
                {
                    label: 'Std Dev (±)',
                    data: stdReturns,
                    backgroundColor: 'rgba(255,255,255,0.04)',
                    borderColor: 'rgba(255,255,255,0.12)',
                    borderWidth: 1,
                    borderRadius: 6,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: { label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%' }
                }
            },
            scales: {
                x: { grid: { display: false } },
                y: {
                    ticks: { callback: v => v + '%', font: { size: 10 } },
                }
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════
// MARKET DETAIL CHARTS
// ═══════════════════════════════════════════════════════════

function renderMarketCharts() {
    const log = demoData?.log || [];
    if (log.length === 0) return;

    const labels = log.map(l => 'M' + l.month);
    const rates = log.map(l => (l.market?.interest_rate || 0.065) * 100);
    const demands = log.map(l => (l.market?.demand || 0.5) * 100);

    // Interest Rate
    const ctxRate = document.getElementById('chart-interest');
    if (ctxRate) {
        if (chartInterest) chartInterest.destroy();
        const grad = ctxRate.getContext('2d').createLinearGradient(0, 0, 0, 210);
        grad.addColorStop(0, 'rgba(167, 139, 250, 0.3)');
        grad.addColorStop(1, 'rgba(167, 139, 250, 0.0)');

        chartInterest = new Chart(ctxRate, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Interest Rate (%)',
                    data: rates,
                    borderColor: COLORS.purple,
                    backgroundColor: grad,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => c.parsed.y.toFixed(2) + '%' } } },
                scales: {
                    x: { ticks: { maxTicksLimit: 6, font: { size: 9 } }, grid: { display: false } },
                    y: { ticks: { callback: v => v.toFixed(1) + '%', font: { size: 9 } } }
                }
            }
        });
    }

    // Demand
    const ctxDemand = document.getElementById('chart-demand');
    if (ctxDemand) {
        if (chartDemand) chartDemand.destroy();
        const grad = ctxDemand.getContext('2d').createLinearGradient(0, 0, 0, 210);
        grad.addColorStop(0, 'rgba(251, 191, 36, 0.3)');
        grad.addColorStop(1, 'rgba(251, 191, 36, 0.0)');

        chartDemand = new Chart(ctxDemand, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Demand Index',
                    data: demands,
                    borderColor: COLORS.amber,
                    backgroundColor: grad,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => c.parsed.y.toFixed(1) + '%' } } },
                scales: {
                    x: { ticks: { maxTicksLimit: 6, font: { size: 9 } }, grid: { display: false } },
                    y: { ticks: { callback: v => v.toFixed(0) + '%', font: { size: 9 } } }
                }
            }
        });
    }

    // Cash Balance
    const ctxCF = document.getElementById('chart-cashflow');
    if (ctxCF) {
        if (chartCashflow) chartCashflow.destroy();
        const cashAmounts = log.map(l => l.cash_after || 0);
        const grad = ctxCF.getContext('2d').createLinearGradient(0, 0, 0, 210);
        grad.addColorStop(0, 'rgba(34, 211, 238, 0.3)');
        grad.addColorStop(1, 'rgba(34, 211, 238, 0.0)');

        chartCashflow = new Chart(ctxCF, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Cash Balance (₹)',
                    data: cashAmounts,
                    borderColor: COLORS.cyan,
                    backgroundColor: grad,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => formatINRFull(c.parsed.y) } } },
                scales: {
                    x: { ticks: { maxTicksLimit: 6, font: { size: 9 } }, grid: { display: false } },
                    y: { ticks: { callback: v => formatINRAxis(v), font: { size: 9 } } }
                }
            }
        });
    }
}

// ═══════════════════════════════════════════════════════════
// ACTION LOG
// ═══════════════════════════════════════════════════════════

function renderActionLog() {
    const container = document.getElementById('action-log');
    const log = demoData?.log || [];

    if (log.length === 0) {
        container.innerHTML = '<p class="placeholder-text">Click "Load Data" or "Simulate" to see agent decisions</p>';
        return;
    }

    const recent = log.slice(-20);
    container.innerHTML = recent.map(entry => {
        const regimeClass = entry.regime?.toLowerCase() || 'stable';
        const actionsHTML = (entry.actions || [])
            .filter(a => a.action !== 'Hold' && a.action !== 'Held')
            .map(a => {
                let cls = '';
                if (a.action === 'Buy') cls = 'buy';
                else if (a.action === 'Sell') cls = 'sell';
                else if (a.action === 'Raise Rent') cls = 'raise';
                else if (a.action === 'Lower Rent') cls = 'lower';
                const status = a.success ? '✓' : '✗';
                return `<span class="log-action-chip ${cls}">S${a.slot} ${a.action} ${status}</span>`;
            }).join('');

        const nw = entry.net_worth ? ' | NW: ' + formatINR(entry.net_worth) : '';

        return `
            <div class="log-entry">
                <span class="log-month">Month ${entry.month}</span>
                <span class="log-regime ${regimeClass}">${entry.regime || 'N/A'}</span>
                <div class="log-actions">
                    ${actionsHTML || '<span class="log-action-chip">All Hold</span>'}
                    <span style="color: var(--text-muted); font-size: 0.62rem; margin-left: 0.5rem">${nw}</span>
                </div>
            </div>
        `;
    }).join('');
}

function toggleLog() {
    const log = document.getElementById('action-log');
    log.classList.toggle('expanded');
}

// ═══════════════════════════════════════════════════════════
// SIMULATION
// ═══════════════════════════════════════════════════════════

function runSimulation() {
    const btn = document.getElementById('btn-run-demo');
    btn.innerHTML = '<span class="btn-icon">⏳</span> Simulating...';

    setTimeout(() => {
        demoData = generateSyntheticDemo();
        if (!evaluationData) useSampleData();
        renderAll();
        btn.innerHTML = '<span class="btn-icon">✅</span> Done!';
        setTimeout(() => {
            btn.innerHTML = '<span class="btn-icon">▶️</span> Simulate';
        }, 2500);
    }, 600);
}

// ═══════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════

function getBestAgent() {
    if (!evaluationData) return null;
    let best = null;
    let bestReturn = -Infinity;
    for (const [name, data] of Object.entries(evaluationData)) {
        if (data.avg_return_pct > bestReturn) {
            bestReturn = data.avg_return_pct;
            best = name;
        }
    }
    return best;
}

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    loadAllData();
});

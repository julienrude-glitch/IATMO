// IATMO - cache des tuiles deja vues (fonds IGN) et de la page elle-meme.
// A deposer a cote de index.html. Sans effet sur la version ouverte comme simple fichier.
const TUILES = "iatmo-tuiles-v1", PAGE = "iatmo-page-v1", MAX = 6000;
let compte = 0;
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", e => e.waitUntil(
  caches.keys().then(ks => Promise.all(ks.filter(k => k.startsWith("iatmo-") && k !== TUILES && k !== PAGE).map(k => caches.delete(k))))
    .then(() => self.clients.claim())));
function elaguer(c){                       // au-dela de MAX tuiles, les plus anciennes sont retirees
  if(++compte % 200) return;
  c.keys().then(ks => { if(ks.length > MAX) ks.slice(0, ks.length - MAX).forEach(k => c.delete(k)); });
}
self.addEventListener("fetch", e => {
  const r = e.request; if(r.method !== "GET") return;
  const u = new URL(r.url);
  // Tuiles IGN (satellite, communes, routes, bati) : le cache d'abord, le reseau sinon
  if(u.hostname === "data.geopf.fr" && (u.pathname.startsWith("/tms/") || u.pathname.startsWith("/wmts"))){
    e.respondWith(caches.open(TUILES).then(c => c.match(r).then(hit => hit || fetch(r).then(rep => {
      if(rep.ok){ c.put(r, rep.clone()); elaguer(c); }
      return rep;
    }))));
    return;
  }
  // La page : le reseau d'abord (pour recevoir les mises a jour), le cache en secours hors ligne
  if(u.origin === self.location.origin && (r.mode === "navigate" || u.pathname.endsWith("/index.html"))){
    e.respondWith(fetch(r).then(rep => { if(rep.ok){ const copie = rep.clone(); caches.open(PAGE).then(c => c.put(r, copie)); } return rep; })
      .catch(() => caches.match(r).then(hit => hit || Response.error())));
  }
});

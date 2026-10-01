// Small ZIP-STORED writer/reader for our bundles. No executable archive content is evaluated.
const enc = new TextEncoder(), dec = new TextDecoder();
export function crc32(data) { let c=0xffffffff; for(const b of data){c^=b; for(let i=0;i<8;i++)c=(c>>>1)^((c&1)?0xedb88320:0);}return (c^0xffffffff)>>>0; }
export function zip(files) {
  const chunks=[], directory=[]; let offset=0;
  for(const [name,text] of Object.entries(files)) {
    const n=enc.encode(name), data=typeof text==='string'?enc.encode(text):text;
    const crc=crc32(data), h=new Uint8Array(30+n.length), v=new DataView(h.buffer);
    v.setUint32(0,0x04034b50,true);v.setUint16(4,20,true);v.setUint16(6,0x800,true);v.setUint32(14,crc,true);v.setUint32(18,data.length,true);v.setUint32(22,data.length,true);v.setUint16(26,n.length,true);h.set(n,30);
    const d=new Uint8Array(46+n.length), dv=new DataView(d.buffer);
    dv.setUint32(0,0x02014b50,true);dv.setUint16(4,20,true);dv.setUint16(6,20,true);dv.setUint16(8,0x800,true);dv.setUint32(16,crc,true);dv.setUint32(20,data.length,true);dv.setUint32(24,data.length,true);dv.setUint16(28,n.length,true);dv.setUint32(42,offset,true);d.set(n,46);
    chunks.push(h,data);directory.push(d);offset+=h.length+data.length;
  }
  const size=directory.reduce((s,d)=>s+d.length,0), end=new Uint8Array(22), v=new DataView(end.buffer);
  v.setUint32(0,0x06054b50,true);v.setUint16(8,directory.length,true);v.setUint16(10,directory.length,true);v.setUint32(12,size,true);v.setUint32(16,offset,true);
  return new Blob([...chunks,...directory,end],{type:'application/zip'});
}
export function unzip(buffer) {
  const data=new Uint8Array(buffer), v=new DataView(buffer), files={}, records=new Map();
  let p=0,total=0,endRecord=-1;
  if(data.length>64*1024*1024)throw new Error('Import the compact report-data.json; archives are limited to 64 MiB.');
  // Require the complete end record, including any bounded ZIP comment.
  for(let i=data.length-22;i>=Math.max(0,data.length-22-65535);i--){
    if(v.getUint32(i,true)===0x06054b50 && i+22+v.getUint16(i+20,true)===data.length){endRecord=i;break;}
  }
  if(endRecord<0)throw new Error('Truncated ZIP: missing end of central directory.');
  const count=v.getUint16(endRecord+10,true),size=v.getUint32(endRecord+12,true),directory=v.getUint32(endRecord+16,true);
  if(v.getUint16(endRecord+4,true)!==0||v.getUint16(endRecord+6,true)!==0||v.getUint16(endRecord+8,true)!==count||count<1||count>1000||directory+size!==endRecord)throw new Error('Invalid or unsupported ZIP directory.');
  while(p<directory){
    if(p+30>directory||v.getUint32(p,true)!==0x04034b50)throw new Error('Truncated archive.');
    const flags=v.getUint16(p+6,true), method=v.getUint16(p+8,true), crc=v.getUint32(p+14,true), length=v.getUint32(p+18,true), n=v.getUint16(p+26,true), extra=v.getUint16(p+28,true);
    if(method!==0 || flags&~0x800)throw new Error('Use the helper-generated uncompressed results.zip or report-data.json.');
    const start=p+30+n+extra, end=start+length;
    if(end>directory || length!==v.getUint32(p+22,true))throw new Error('Truncated or invalid archive.');
    const name=dec.decode(data.subarray(p+30,p+30+n));
    if(!name || name.split('/').some(v=>v==='..') || name.startsWith('/') || /^[a-z]:/i.test(name) || name.includes('\\') || name.includes('\0') || Object.hasOwn(files,name) || records.size>=count)throw new Error('Unsafe or duplicate archive entry.');
    const bytes=data.subarray(start,end);total+=length;
    if(total>64*1024*1024 || crc32(bytes)!==crc)throw new Error('Archive checksum or size check failed.');
    records.set(p,{name,flags,crc,length});
    Object.defineProperty(files,name,{value:dec.decode(bytes),enumerable:true});p=end;
  }
  if(records.size!==count)throw new Error('Invalid ZIP entry count.');
  const seen=new Set();
  for(let i=0;i<count;i++){
    if(p+46>endRecord||v.getUint32(p,true)!==0x02014b50)throw new Error('Truncated ZIP central directory.');
    const n=v.getUint16(p+28,true),extra=v.getUint16(p+30,true),comment=v.getUint16(p+32,true),offset=v.getUint32(p+42,true);
    const end=p+46+n+extra+comment,record=records.get(offset);
    if(end>endRecord||!record||seen.has(offset)||v.getUint16(p+34,true)!==0||v.getUint16(p+8,true)!==record.flags||v.getUint16(p+10,true)!==0||v.getUint32(p+16,true)!==record.crc||v.getUint32(p+20,true)!==record.length||v.getUint32(p+24,true)!==record.length||dec.decode(data.subarray(p+46,p+46+n))!==record.name)throw new Error('ZIP central directory disagrees with archive records.');
    seen.add(offset);p=end;
  }
  if(p!==endRecord)throw new Error('Invalid ZIP central directory size.');
  return files;
}

this.imagePreview = function(){			
	var xOffset = 20;
	var yOffset = 10;
	var w = $("div.wrapper").width();
	var h = $("div.wrapper").height();
	
	$("a.preview").hover(function(e){		
		this.t = this.title;
		this.title = "";	
		var c = (this.t != "") ? "<br/>" + this.t : "";
		$("div.wrapper").append("<p id='preview' class='col-sm-5 col-lg-5 col-md-5 thumbnail'><img src='"+ this.href +"' alt='Image preview' />"+ c +"</p>");	

		var imgW = $("#preview").width();
		var imgH = $("#preview").height();
		if (e.pageX + imgW + 20 > w) xOffset = -imgW - 20;
		else xOffset = 20;
		if (e.pageY + imgH + 10 > h) yOffset = -imgH - 10;
		else yOffset = 10;
		
		$("#preview")
			.css("top",(e.pageY + yOffset) + "px")
			.css("left",(e.pageX + xOffset) + "px")
			.fadeIn("fast");						
    },
	function(){
		this.title = this.t;	
		$("#preview").remove();
    });	
	$("a.preview").mousemove(function(e){
		var imgW = $("#preview").width();
		var imgH = $("#preview").height();
		if (e.pageX + imgW + 20 > w) xOffset = -imgW - 20;
		else xOffset = 20;
		if (e.pageY + imgH + 10 > h) yOffset = -imgH - 10;
		else yOffset = 10;
		
		$("#aaa").text(e.pageX + " - " + e.pageY + " - " + w + ":" + h + " - " + imgW + ":" + imgH);
		$("#preview")
			.css("top",(e.pageY + yOffset) + "px")
			.css("left",(e.pageX + xOffset) + "px");
	});			
};

// starting the script on page load
$(document).ready(function(){
	imagePreview();
});

function blacklistClick(){
	document.getElementById("Blacklist-div").style.display = "block";
	document.getElementById("Unknown-div").style.display = "none";
	document.getElementById("Whitelist-div").style.display = "none";

	document.getElementById("bl-treeview").className = "active treeview";
	document.getElementById("uk-treeview").className = "treeview";
	document.getElementById("wl-treeview").className = "treeview";

	getBlacklistImgList();
}

function unknownClick(){
	document.getElementById("Blacklist-div").style.display = "none";
	document.getElementById("Unknown-div").style.display = "block";
	document.getElementById("Whitelist-div").style.display = "none";

	document.getElementById("bl-treeview").className = "treeview";
	document.getElementById("uk-treeview").className = "active treeview";
	document.getElementById("wl-treeview").className = "treeview";
}

function whitelistClick(){
	document.getElementById("Blacklist-div").style.display = "none";
	document.getElementById("Unknown-div").style.display = "none";
	document.getElementById("Whitelist-div").style.display = "block";

	document.getElementById("bl-treeview").className = "treeview";
	document.getElementById("uk-treeview").className = "treeview";
	document.getElementById("wl-treeview").className = "active treeview";
}

var bl_data;
function getBlacklistImgList(){

	$.ajax({
		url: window.location.pathname+'rest/ws/listvideo/bl',
		type: 'GET',
		async: true,
		success: function (data) {
//			alert(data);
			bl_data=data;

			var obj = JSON.parse(data);
			console.log(obj.videos.length);	
			$(".blacklist-content").empty();
			for(var i = 0 ; i < obj.videos.length ; i++){
				var video_name = obj.videos[i];
				console.log(video_name);
			}
		},
		cache: false,
		contentType: false,
		processData: false,
	});
}

function addBlacklistItem(imgPath){
	var item = imgPath.split("_");
	var name = item[1].replace("-"," ");
	var uid = item[2];
	var fullImageName = items[3];
	var time = fullImageName.replace(".jpg","");

	var host = window.location.pathname+'rest/ws/img/getImage?img=';
	
	var cell = document.createElement("div");
	cell.className="col-sm-3 col-lg-3 col-md-3";
	
	var thumbnail_div = document.createElement("div");
	thumbnail_div.className="thumbnail";
	
	var img = document.createElement("img");
	img.src=host+imgPath;
	
	var caption_div = document.createElement("div");
	div.className="caption";
	
	var h4 = document.createElement("h4");
	var name_a = document.createElement("a");
	name_a.href="#";
	name_a.innerHTML=name;
	h4.appendChild(name_a);
	
}

function parseTimestamp(unix_tm) {
    var dt = new Date(unix_tm*1000);
    var year = dt.getFullYear();
    var month = dt.getMonth()+1;
    var date = dt.getDate();
    var hour = dt.getHours();
    var min = dt.getMinutes();
    var sec = dt.getSeconds();
    
    return date+'/'+month+'/'+year+' '+hour+':'+min+':'+sec;
}
